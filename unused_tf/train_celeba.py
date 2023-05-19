import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf
# Set memory growth for GPU
gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)
import datasets
import models
from pathlib import Path
import tqdm
from contextlib import nullcontext
class Trainer():
    def __init__(
            self,
            dataset_name,
            dataset_kwargs,
            generator_name,
            generator_kwargs,
            discriminator_name,
            discriminator_kwargs,
            gan_loss_name,
            gan_loss_kwargs,
            noise_size,
            name,
            lr=None,
            distributed = False,
            checkpoint_restore_path=None,
            save_period=500,
            lr_schedule_name=None,
            lr_schedule_kwargs=None,
            lr_generator=None,
            lr_discriminator=None,
            **kwargs
    ):
        """
        Parameters
        ----------
        dataset_name : str
            dataset name
        dataset_kwargs : dict
            dataset kwargs
        generator_name : str
            generator name
        generator_kwargs : dict
            generator kwargs
        discriminator_name : str
            discriminator name
        discriminator_kwargs : dict
            discriminator kwargs
        gan_loss_name : str
            gan loss name
        gan_loss_kwargs : dict
            gan loss kwargs
        noise_size : tuple
            noise size (does not include batch size)
        name : str
            name of the trainer
        lr : float
            learning rate (ignored if lr_schedule_name is not None or lr_generator is not None or lr_discriminator is not None)
        distributed : bool  
            whether to use distributed training
        checkpoint_restore_path : str
            checkpoint restore path
        save_period : int
            save period
        lr_schedule_name : str
            learning rate schedule name
        lr_schedule_kwargs : dict
            learning rate schedule kwargs
        lr_generator : float
            learning rate for generator (ignored if lr_schedule_name is not None)
        lr_discriminator : float
            learning rate for discriminator (ignored if lr_schedule_name is not None)
        """
        self.save_period = save_period
        self.noise_size = noise_size

        # assert 'batch_size' is in dataset_kwargs
        assert 'batch_size' in dataset_kwargs
        #* batch_size in dataset_kwargs is the global batch size
        train_batch_size = dataset_kwargs['batch_size']

        # if 'normalize' is in dataset_kwargs, set self.normalize
        if 'normalize' in dataset_kwargs:
            self.normalize = dataset_kwargs['normalize']
        else:
            self.normalize = None

        if distributed:
            self.strategy = tf.distribute.MirroredStrategy()
        else :
            self.strategy = None
        self.train_dataset, n = getattr(datasets, dataset_name)(**dataset_kwargs)
        if distributed:
            self.train_dataset = self.strategy.experimental_distribute_dataset(self.train_dataset)
        
        self.val_seed = tf.random.normal((16, *self.noise_size))

        self.steps = n // train_batch_size

        with self.strategy.scope() if distributed else nullcontext():

            generator_kwargs['name'] = 'generator'
            self.generator = getattr(models, generator_name)(**generator_kwargs)

            discriminator_kwargs['name'] = 'discriminator'
            self.discriminator = getattr(models, discriminator_name)(**discriminator_kwargs)

            if distributed:
                gan_loss_kwargs['reduction'] = tf.keras.losses.Reduction.NONE
            gan_loss_obj = getattr(tf.keras.losses, gan_loss_name)(**gan_loss_kwargs)
            if distributed:
                # distribution loss function
                def distributed_gan_loss(y_true, y_pred):
                    # y_true shape: (batch_size, ...)
                    per_example_loss = gan_loss_obj(y_true, y_pred)
                    # shape (batch_size, ...) : tf.keras.losses.Reduction.NONE reduces the last dimension
                    # Flatten except batch dimension
                    batch_size = tf.shape(y_true)[0]
                    per_example_loss = tf.reshape(per_example_loss, (batch_size, -1))
                    per_example_loss = tf.reduce_mean(per_example_loss, axis=1)
                    #* Note: tf.nn.compute_average_loss internally sums along all dimensions,
                    #* while tf.keras.losses.Reduction.NONE only reduces the last dimension.
                    #* If not manually reduce_mean, the loss will be scaled by the number of pixels.
                    return tf.nn.compute_average_loss(per_example_loss, global_batch_size=train_batch_size)
                self.gan_loss_obj = distributed_gan_loss
            else:
                self.gan_loss_obj = gan_loss_obj

            # assert at least one of lr and lr_schedule_name is not None
            assert lr is not None or lr_schedule_name is not None or \
                (lr_generator is not None and lr_discriminator is not None)

            if lr_schedule_name is not None:
                lr = getattr(tf.keras.optimizers.schedules, lr_schedule_name)(**lr_schedule_kwargs)
                self.lr_generator = lr
                self.lr_discriminator = lr
            elif lr_generator is not None and lr_discriminator is not None:
                self.lr_generator = lr_generator
                self.lr_discriminator = lr_discriminator
            else:
                self.lr_generator = lr
                self.lr_discriminator = lr

            self.generator_optimizer = tf.keras.optimizers.Adam(self.lr_generator, beta_1=0.5)
            self.discriminator_optimizer = tf.keras.optimizers.Adam(self.lr_discriminator, beta_1=0.5)

            self.name = name
            self.log_dir = Path('logs') / self.name
            self.checkpoint_dir = self.log_dir / 'checkpoints'
            self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
            self.checkpoint_prefix = str(self.checkpoint_dir / 'ckpt')
            self.checkpoint = tf.train.Checkpoint(
                generator=self.generator,
                discriminator=self.discriminator,
                generator_optimizer=self.generator_optimizer,
                discriminator_optimizer=self.discriminator_optimizer,
            )

            if checkpoint_restore_path is not None:
                self.checkpoint.restore(checkpoint_restore_path)
                print(f'Restored from {checkpoint_restore_path}')
            self.writer = tf.summary.create_file_writer(str(self.log_dir))
            self.writer.set_as_default()

            self.train_gen_loss = tf.keras.metrics.Mean(name='train_gen_loss')
            self.train_disc_loss = tf.keras.metrics.Mean(name='train_disc_loss')

        

    def discriminator_loss(self, real, generated):
        real_loss = self.gan_loss_obj(tf.ones_like(real), real)
        generated_loss = self.gan_loss_obj(tf.zeros_like(generated), generated)
        total_disc_loss = real_loss + generated_loss
        return total_disc_loss
    
    def generator_loss(self, generated):
        return self.gan_loss_obj(tf.ones_like(generated), generated)
    
    
    @tf.function
    def train_step(self, images):
        batch_size = tf.shape(images)[0:1]
        noise_size = tf.concat([batch_size, self.noise_size], axis=0)
        noise = tf.random.normal(noise_size)
        with tf.GradientTape(persistent=True) as tape:
            generated_images = self.generator(noise, training=True)

            real_output = self.discriminator(images, training=True)
            fake_output = self.discriminator(generated_images, training=True)

            gen_loss = self.generator_loss(fake_output)
            disc_loss = self.discriminator_loss(real_output, fake_output)

        gen_gradients = tape.gradient(gen_loss, self.generator.trainable_variables)
        disc_gradients = tape.gradient(disc_loss, self.discriminator.trainable_variables)

        self.generator_optimizer.apply_gradients(zip(gen_gradients, self.generator.trainable_variables))
        self.discriminator_optimizer.apply_gradients(zip(disc_gradients, self.discriminator.trainable_variables))

        self.train_gen_loss.update_state(gen_loss)
        self.train_disc_loss.update_state(disc_loss)

    @tf.function
    def distributed_train_step(self, images):
        self.strategy.run(self.train_step, args=(images,))

    def tb_record_image(self, step):
        generated_images = self.generator(self.val_seed, training=False)
        if self.normalize == 'tanh':
            generated_images = (generated_images + 1) / 2
        tf.summary.image('generated_images', generated_images, step=step, max_outputs=16)

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
            for images in step_tqdm:
                if self.strategy is not None:
                    self.distributed_train_step(images)
                else:
                    self.train_step(images)
                if step_counter % self.save_period == 0:
                    self.tb_record_image(step_counter)
                    tf.summary.scalar('train_gen_loss', self.train_gen_loss.result(), step=step_counter)
                    tf.summary.scalar('train_disc_loss', self.train_disc_loss.result(), step=step_counter)
                    self.writer.flush()
                    self.train_gen_loss.reset_states()
                    self.train_disc_loss.reset_states()
                    # save model
                    self.checkpoint.save(self.checkpoint_prefix)
                step_counter += 1
            
            
            
            

if __name__ == '__main__':
    import json
    with open('configs/config_celeba.json', 'r') as f:
        config = json.load(f)
    trainer = Trainer(**config)
    with open(trainer.log_dir/'config.json', 'w') as f:
        json.dump(config, f, indent=4)
    trainer.fit(config['epochs'])
    