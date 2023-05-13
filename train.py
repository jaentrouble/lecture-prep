import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf
from datasets import CNCPairDataset, ContrastSample
import models
from pathlib import Path
import tqdm
from contextlib import nullcontext
class Trainer():
    def __init__(
            self,
            train_contrast_dir_list,
            train_non_contrast_dir_list,
            val_contrast_dir_list,
            generator_name,
            generator_kwargs,
            discriminator_name,
            discriminator_kwargs,
            gan_loss_name,
            gan_loss_kwargs,
            cycle_loss_name,
            cycle_loss_kwargs,
            cycle_loss_lambda,
            name,
            lr=None,
            train_batch_size=1,
            distributed = False,
            batch_size_per_replica = 1,
            resize=None,
            checkpoint_restore_path=None,
            save_period=500,
            lr_schedule_name=None,
            lr_schedule_kwargs=None,
            **kwargs
    ):
        """
        Parameters
        ----------
        train_contrast_dir_list : list of str
            List of directories containing contrast images for training.
        train_non_contrast_dir_list : list of str
            List of directories containing non-contrast images for training.
        val_contrast_dir_list : list of str
            List of directories containing contrast images for validation.
        generator_name : str
            Name of the generator model.
        generator_kwargs : dict
            Keyword arguments for the generator model.
        discriminator_name : str
            Name of the discriminator model.
        discriminator_kwargs : dict
            Keyword arguments for the discriminator model.
        gan_loss_name : str
            Name of the GAN loss function.
            This is used for both the generator and discriminator loss.
        gan_loss_kwargs : dict
            Keyword arguments for the GAN loss function.
        cycle_loss_name : str
            This is used for both the cycle loss and identity loss.
            Name of the cycle loss function.
        cycle_loss_kwargs : dict
            Keyword arguments for the cycle loss function.
        cycle_loss_lambda : float
            Weight of the cycle loss.
        lr : float
            Learning rate. Ignored if lr_schedule_name is not None.
        name : str
            Name of the experiment.
        train_batch_size : int
            Batch size. (Ignored if distributed is True)
        distributed : bool
            Whether to use distributed training.
        batch_size_per_replica : int
            Batch size per replica. (Used only if distributed is True)
        resize : tuple of int
            Size to resize the images to.
        checkpoint_restore_path : str
            Path to the checkpoint to restore.
        save_period : int
            Period (steps) to save the checkpoint.
        lr_schedule_name : str
            Name of the learning rate schedule. (Optional)
        lr_schedule_kwargs : dict
            Keyword arguments for the learning rate schedule. (Optional)
        """
        self.save_period = save_period

        if distributed:
            self.strategy = tf.distribute.MirroredStrategy()
            train_batch_size = batch_size_per_replica * self.strategy.num_replicas_in_sync
        else :
            self.strategy = None
        self.train_dataset, n = CNCPairDataset(
            contrast_dir_list=train_contrast_dir_list,
            non_contrast_dir_list=train_non_contrast_dir_list,
            shuffle=True,
            batch_size=train_batch_size,
            resize=resize
        )
        if distributed:
            self.train_dataset = self.strategy.experimental_distribute_dataset(self.train_dataset)

        self.steps = n // train_batch_size + (n % train_batch_size != 0)
        self.val_sample = ContrastSample(
            val_contrast_dir_list,
            resize=resize)
        self.val_batch_size = self.val_sample.shape[0]

        with self.strategy.scope() if distributed else nullcontext():

            generator_kwargs['name'] = 'generator_g'
            self.generator_g = getattr(models, generator_name)(**generator_kwargs)
            generator_kwargs['name'] = 'generator_f'
            self.generator_f = getattr(models, generator_name)(**generator_kwargs)

            self.discriminator_x = getattr(models, discriminator_name)(**discriminator_kwargs)
            self.discriminator_y = getattr(models, discriminator_name)(**discriminator_kwargs)

            if distributed:
                gan_loss_kwargs['reduction'] = tf.keras.losses.Reduction.NONE
                cycle_loss_kwargs['reduction'] = tf.keras.losses.Reduction.NONE
            gan_loss_obj = getattr(tf.keras.losses, gan_loss_name)(**gan_loss_kwargs)
            cycle_loss_obj = getattr(tf.keras.losses, cycle_loss_name)(**cycle_loss_kwargs)
            if distributed:
                # distribution loss function
                def distributed_gan_loss(y_true, y_pred):
                    # Expects PatchGAN like input
                    # y_true shape: (batch_size, height, width, 1)
                    per_example_loss = gan_loss_obj(y_true, y_pred)
                    # shape (batch_size, height, width) : tf.keras.losses.Reduction.NONE
                    per_example_loss = tf.reduce_mean(per_example_loss, axis=[1, 2])
                    #* Note: tf.nn.compute_average_loss internally sums along all dimensions,
                    #* while tf.keras.losses.Reduction.NONE only reduces the last dimension.
                    #* If not manually reduce_mean, the loss will be scaled by the number of pixels.
                    return tf.nn.compute_average_loss(per_example_loss, global_batch_size=train_batch_size)
                def distributed_cycle_loss(y_true, y_pred):
                    per_example_loss = cycle_loss_obj(y_true, y_pred)
                    # shape (batch_size, height, width) : tf.keras.losses.Reduction.NONE
                    per_example_loss = tf.reduce_mean(per_example_loss, axis=[1, 2])
                    return tf.nn.compute_average_loss(per_example_loss, global_batch_size=train_batch_size)
                self.gan_loss_obj = distributed_gan_loss
                self.cycle_loss_obj = distributed_cycle_loss
            else:
                self.gan_loss_obj = gan_loss_obj
                self.cycle_loss_obj = cycle_loss_obj
            self.cycle_loss_lambda = cycle_loss_lambda

            # assert at least one of lr and lr_schedule_name is not None
            assert lr is not None or lr_schedule_name is not None

            if lr_schedule_name is not None:
                self.lr = getattr(tf.keras.optimizers.schedules, lr_schedule_name)(**lr_schedule_kwargs)
            else:
                self.lr = lr
            self.generator_g_optimizer = tf.keras.optimizers.Adam(self.lr, beta_1=0.5)
            self.generator_f_optimizer = tf.keras.optimizers.Adam(self.lr, beta_1=0.5)
            self.discriminator_x_optimizer = tf.keras.optimizers.Adam(self.lr, beta_1=0.5)
            self.discriminator_y_optimizer = tf.keras.optimizers.Adam(self.lr, beta_1=0.5)

            self.name = name
            self.log_dir = Path('logs') / self.name
            self.checkpoint_dir = self.log_dir / 'checkpoints'
            self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
            self.checkpoint_prefix = str(self.checkpoint_dir / 'ckpt')
            self.checkpoint = tf.train.Checkpoint(
                generator_g=self.generator_g,
                generator_f=self.generator_f,
                discriminator_x=self.discriminator_x,
                discriminator_y=self.discriminator_y,
                generator_g_optimizer=self.generator_g_optimizer,
                generator_f_optimizer=self.generator_f_optimizer,
                discriminator_x_optimizer=self.discriminator_x_optimizer,
                discriminator_y_optimizer=self.discriminator_y_optimizer,
            )

            if checkpoint_restore_path is not None:
                self.checkpoint.restore(checkpoint_restore_path)
                print(f'Restored from {checkpoint_restore_path}')
            self.writer = tf.summary.create_file_writer(str(self.log_dir))
            self.writer.set_as_default()

            self.train_gen_g_loss = tf.keras.metrics.Mean(name='train_gen_g_loss')
            self.train_gen_f_loss = tf.keras.metrics.Mean(name='train_gen_f_loss')
            self.train_disc_x_loss = tf.keras.metrics.Mean(name='train_disc_x_loss')
            self.train_disc_y_loss = tf.keras.metrics.Mean(name='train_disc_y_loss')

        

    def discriminator_loss(self, real, generated):
        real_loss = self.gan_loss_obj(tf.ones_like(real), real)
        generated_loss = self.gan_loss_obj(tf.zeros_like(generated), generated)
        total_disc_loss = real_loss + generated_loss
        return total_disc_loss * 0.5
    
    def generator_loss(self, generated):
        return self.gan_loss_obj(tf.ones_like(generated), generated)
    
    def cycle_loss(self, real_image, cycled_image):
        return self.cycle_loss_lambda * self.cycle_loss_obj(real_image, cycled_image)
    
    def identity_loss(self, real_image, same_image):
        return self.cycle_loss_lambda * 0.5 * self.cycle_loss_obj(real_image, same_image)
    
    @tf.function
    def train_step(self, contrast, non_contrast):
        # X is contrast, Y is non-contrast.
        with tf.GradientTape(persistent=True) as tape:
            # Generator G translates X -> Y
            # Generator F translates Y -> X.
            fake_y = self.generator_g(contrast, training=True)
            cycled_x = self.generator_f(fake_y, training=True)

            fake_x = self.generator_f(non_contrast, training=True)
            cycled_y = self.generator_g(fake_x, training=True)

            # same_x and same_y are used for identity loss.
            same_x = self.generator_f(contrast, training=True)
            same_y = self.generator_g(non_contrast, training=True)

            disc_real_x = self.discriminator_x(contrast, training=True)
            disc_real_y = self.discriminator_y(non_contrast, training=True)

            disc_fake_x = self.discriminator_x(fake_x, training=True)
            disc_fake_y = self.discriminator_y(fake_y, training=True)

            # calculate the loss
            gen_g_loss = self.generator_loss(disc_fake_y)
            gen_f_loss = self.generator_loss(disc_fake_x)

            total_cycle_loss = self.cycle_loss(contrast, cycled_x) + self.cycle_loss(non_contrast, cycled_y)

            # Total generator loss = adversarial loss + cycle loss
            total_gen_g_loss = gen_g_loss + total_cycle_loss + self.identity_loss(contrast, same_x)
            total_gen_f_loss = gen_f_loss + total_cycle_loss + self.identity_loss(non_contrast, same_y)

            disc_x_loss = self.discriminator_loss(disc_real_x, disc_fake_x)
            disc_y_loss = self.discriminator_loss(disc_real_y, disc_fake_y)

        # Calculate the gradients for generator and discriminator
        generator_g_gradients = tape.gradient(total_gen_g_loss, self.generator_g.trainable_variables)
        generator_f_gradients = tape.gradient(total_gen_f_loss, self.generator_f.trainable_variables)

        discriminator_x_gradients = tape.gradient(disc_x_loss, self.discriminator_x.trainable_variables)
        discriminator_y_gradients = tape.gradient(disc_y_loss, self.discriminator_y.trainable_variables)

        # Apply the gradients to the optimizer
        self.generator_g_optimizer.apply_gradients(zip(generator_g_gradients, 
                                                       self.generator_g.trainable_variables))

        self.generator_f_optimizer.apply_gradients(zip(generator_f_gradients, 
                                                       self.generator_f.trainable_variables))

        self.discriminator_x_optimizer.apply_gradients(zip(discriminator_x_gradients, 
                                                           self.discriminator_x.trainable_variables))
        self.discriminator_y_optimizer.apply_gradients(zip(discriminator_y_gradients,
                                                              self.discriminator_y.trainable_variables))
        
        self.train_gen_g_loss.update_state(total_gen_g_loss)
        self.train_gen_f_loss.update_state(total_gen_f_loss)
        self.train_disc_x_loss.update_state(disc_x_loss)
        self.train_disc_y_loss.update_state(disc_y_loss)

    @tf.function
    def distributed_train_step(self, contrast, non_contrast):
        self.strategy.run(self.train_step, args=(contrast, non_contrast))

    def tb_record_image(self, contrast, noncontrast, epoch, prefix='train', max_outputs=1,
                        forward_only=False):
        # if contrast and noncontrast is per_replica, get the first replica value of contrast and noncontrast
        contrast = contrast.values[0] if isinstance(contrast, tf.distribute.DistributedValues) else contrast
        noncontrast = noncontrast.values[0] if isinstance(noncontrast, tf.distribute.DistributedValues) else noncontrast
        # x is contrast, y is noncontrast
        if forward_only:
            fake_y = self.generator_g(contrast, training=False)
        else:
            fake_y = self.generator_g(contrast, training=False)
            cycled_x = self.generator_f(fake_y, training=False)

            fake_x = self.generator_f(noncontrast, training=False)
            cycled_y = self.generator_g(fake_x, training=False)

            same_x = self.generator_f(contrast, training=False)
            same_y = self.generator_g(noncontrast, training=False)

            disc_real_x = self.discriminator_x(contrast, training=False)
            disc_real_y = self.discriminator_y(noncontrast, training=False)

            disc_fake_x = self.discriminator_x(fake_x, training=False)
            disc_fake_y = self.discriminator_y(fake_y, training=False)

        if prefix == 'train':
            tf.summary.image(f'{prefix}_contrast', contrast, step=epoch, max_outputs=max_outputs)
            tf.summary.image(f'{prefix}_noncontrast', noncontrast, step=epoch, max_outputs=max_outputs)
            tf.summary.image(f'{prefix}_fake_y', fake_y, step=epoch, max_outputs=max_outputs)
            tf.summary.image(f'{prefix}_cycled_x', cycled_x, step=epoch, max_outputs=max_outputs)
            tf.summary.image(f'{prefix}_fake_x', fake_x, step=epoch, max_outputs=max_outputs)
            tf.summary.image(f'{prefix}_cycled_y', cycled_y, step=epoch, max_outputs=max_outputs)
            tf.summary.image(f'{prefix}_same_x', same_x, step=epoch, max_outputs=max_outputs)
            tf.summary.image(f'{prefix}_same_y', same_y, step=epoch, max_outputs=max_outputs)
            tf.summary.image(f'{prefix}_disc_real_x', disc_real_x, step=epoch, max_outputs=max_outputs)
            tf.summary.image(f'{prefix}_disc_real_y', disc_real_y, step=epoch, max_outputs=max_outputs)
            tf.summary.image(f'{prefix}_disc_fake_x', disc_fake_x, step=epoch, max_outputs=max_outputs)
            tf.summary.image(f'{prefix}_disc_fake_y', disc_fake_y, step=epoch, max_outputs=max_outputs)
        else:
            if forward_only:
                tf.summary.image(f'{prefix}_contrast', contrast, step=epoch, max_outputs=max_outputs)
                tf.summary.image(f'{prefix}_fake_y', fake_y, step=epoch, max_outputs=max_outputs)
            else:
                tf.summary.image(f'{prefix}_contrast', contrast, step=epoch, max_outputs=max_outputs)
                tf.summary.image(f'{prefix}_fake_y', fake_y, step=epoch, max_outputs=max_outputs)
                tf.summary.image(f'{prefix}_noncontrast', noncontrast, step=epoch, max_outputs=max_outputs)
                tf.summary.image(f'{prefix}_fake_x', fake_x, step=epoch, max_outputs=max_outputs)

    def fit(self, epochs):
        step_counter = 0
        for epoch in tqdm.trange(epochs, desc='Epochs', ncols=80, unit='epoch',
                                 leave=True):
            step_tqdm = tqdm.tqdm(self.train_dataset,
                                    ncols=80,
                                    desc='Steps',
                                    unit='step',
                                    leave=False,
                                    total=self.steps)
            for contrast, noncontrast in step_tqdm:
                if self.strategy is not None:
                    self.distributed_train_step(contrast, noncontrast)
                else:
                    self.train_step(contrast, noncontrast)
                if step_counter % self.save_period == 0:
                    self.tb_record_image(contrast, noncontrast, step_counter, prefix='train')
                    self.tb_record_image(self.val_sample, None, epoch, prefix='val',
                                         max_outputs=self.val_batch_size, forward_only=True)
                    tf.summary.scalar('train_gen_g_loss', self.train_gen_g_loss.result(), step=step_counter)
                    tf.summary.scalar('train_gen_f_loss', self.train_gen_f_loss.result(), step=step_counter)
                    tf.summary.scalar('train_disc_x_loss', self.train_disc_x_loss.result(), step=step_counter)
                    tf.summary.scalar('train_disc_y_loss', self.train_disc_y_loss.result(), step=step_counter)
                    self.train_gen_g_loss.reset_states()
                    self.train_gen_f_loss.reset_states()
                    self.train_disc_x_loss.reset_states()
                    self.train_disc_y_loss.reset_states()
                    # save model
                    self.checkpoint.save(self.checkpoint_prefix)
                step_counter += 1
            
            
            
            

if __name__ == '__main__':
    import json
    with open('configs/config_cycle.json', 'r') as f:
        config = json.load(f)
    trainer = Trainer(**config)
    with open(trainer.log_dir/'config.json', 'w') as f:
        json.dump(config, f, indent=4)
    trainer.fit(config['epochs'])
    