import torch
from torch import nn
import datasets
import models
from pathlib import Path
import tqdm
from torch.utils.tensorboard import SummaryWriter
import torchmetrics as tm

class Trainer():
    def __init__(
        self,
        epochs,
        batch_size,
        dataset_name,
        dataset_kwargs,
        generator_name,
        generator_kwargs,
        discriminator_name,
        discriminator_kwargs,
        gan_loss_name,
        gan_loss_kwargs,
        name,
        lr=None,
        distributed = True,
        lr_schedule_name=None,
        lr_schedule_kwargs=None,
        lr_generator=None,
        lr_discriminator=None,
        **kwargs
    ):
        assert 'noise_size' in generator_kwargs
        self.noise_size = generator_kwargs['noise_size']

        self.val_seed = torch.randn(16, *self.noise_size)

        if 'normalize' in dataset_kwargs:
            self.normalize = dataset_kwargs['normalize']
        else:
            self.normalize = None

        self.epochs = epochs
        self.batch_size = batch_size
        self._build_dataset(dataset_name, dataset_kwargs)
        self._build_models(
            generator_name,
            generator_kwargs,
            discriminator_name,
            discriminator_kwargs,
            distributed,
        )
        self._build_loss(gan_loss_name, gan_loss_kwargs)
        self._build_optimizer(
            lr,
            lr_schedule_name,
            lr_schedule_kwargs,
            lr_generator,
            lr_discriminator,
        )
        self._build_logger(name)

    def _build_dataset(self, dataset_name, dataset_kwargs):
        self.dataset = getattr(datasets, dataset_name)(**dataset_kwargs)
        self.dataloader = torch.utils.data.DataLoader(
            self.dataset,
            batch_size=self.batch_size,
            shuffle=True,
        )

    def _build_models(
        self,
        generator_name,
        generator_kwargs,
        discriminator_name,
        discriminator_kwargs,
        distributed,
    ):
        self.generator = getattr(models, generator_name)(**generator_kwargs)
        self.discriminator = getattr(models, discriminator_name)(**discriminator_kwargs)
        if distributed:
            self.generator = nn.DataParallel(self.generator)
            self.discriminator = nn.DataParallel(self.discriminator)
        self.generator = self.generator.to('cuda')
        self.discriminator = self.discriminator.to('cuda')

    def _build_loss(self, gan_loss_name, gan_loss_kwargs):
        self.gan_loss_obj = getattr(nn, gan_loss_name)(**gan_loss_kwargs)

    def _build_optimizer(
        self,
        lr,
        lr_schedule_name,
        lr_schedule_kwargs,
        lr_generator,
        lr_discriminator,
    ):
        if lr_generator is None:
            lr_generator = lr
        if lr_discriminator is None:
            lr_discriminator = lr
        self.optimizer_generator = torch.optim.Adam(
            self.generator.parameters(),
            lr=lr_generator,
        )
        self.optimizer_discriminator = torch.optim.Adam(
            self.discriminator.parameters(),
            lr=lr_discriminator,
        )
        if lr_schedule_name is not None:
            self.lr_scheduled = True
            self.lr_schedule_generator = getattr(
                torch.optim.lr_scheduler,
                lr_schedule_name,
            )(
                self.optimizer_generator,
                **lr_schedule_kwargs,
            )
            self.lr_schedule_discriminator = getattr(
                torch.optim.lr_scheduler,
                lr_schedule_name,
            )(
                self.optimizer_discriminator,
                **lr_schedule_kwargs,
            )
        else:
            self.lr_scheduled = False

    def _build_logger(self, name):
        self.log_dir = Path('logs') / name
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.writer = SummaryWriter(self.log_dir)
        self.save_dir = self.log_dir / 'checkpoints'
        self.save_dir.mkdir(parents=True, exist_ok=True)

    def _save_checkpoint(self, epoch):
        torch.save(
            self.generator.state_dict(),
            self.save_dir / f'generator_{epoch}.pt',
        )
        torch.save(
            self.discriminator.state_dict(),
            self.save_dir / f'discriminator_{epoch}.pt',
        )

    def _log(self, epoch, loss_g, loss_d):
        self.writer.add_scalar('Loss/generator', loss_g, epoch)
        self.writer.add_scalar('Loss/discriminator', loss_d, epoch)
        # val image
        self.generator.eval()
        with torch.no_grad():
            val_images = self.generator(self.val_seed)
        self.generator.train()
        val_images = val_images.cpu()
        if self.normalize == 'tanh':
            val_images = (val_images + 1.) / 2.
        self.writer.add_images('Val images', val_images, epoch)

    def _discriminator_loss(self, real, fake):
        real_loss = self.gan_loss_obj(real, torch.ones_like(real))
        fake_loss = self.gan_loss_obj(fake, torch.zeros_like(fake))
        return real_loss + fake_loss
    
    def _generator_loss(self, fake):
        return self.gan_loss_obj(fake, torch.ones_like(fake))
    
    def _train_step(self, images):
        noise = torch.randn(images.shape[0], *self.noise_size).to('cuda')
        fake_images = self.generator(noise)
        loss_d = self._discriminator_loss(
            self.discriminator(images),
            self.discriminator(fake_images.detach()),
        )
        self.optimizer_discriminator.zero_grad()
        loss_d.backward()
        self.optimizer_discriminator.step()

        loss_g = self._generator_loss(self.discriminator(fake_images))
        self.optimizer_generator.zero_grad()
        loss_g.backward()
        self.optimizer_generator.step()

        return loss_d, loss_g

    def _train_epoch(self, epoch):
        loss_d_epoch = tm.MeanMetric().to('cuda')
        loss_g_epoch = tm.MeanMetric().to('cuda')
        for images in tqdm.tqdm(
            self.dataloader,
            desc=f'Epoch {epoch}',
            leave=False,
            ncols=80,
        ):
            images = images.to('cuda')
            loss_d, loss_g = self._train_step(images)
            loss_d_epoch.update(loss_d)
            loss_g_epoch.update(loss_g)
        return loss_d_epoch.compute(), loss_g_epoch.compute()
    
    def fit(self):
        for epoch in tqdm.trange(
            self.epochs,
            desc='Training',
            leave=True,
            ncols=80,
            unit='epoch',
        ):
            loss_d, loss_g = self._train_epoch(epoch)
            if self.lr_scheduled:
                self.lr_schedule_generator.step()
                self.lr_schedule_discriminator.step()
            self._log(epoch, loss_g, loss_d)
            self._save_checkpoint(epoch)

if __name__ == '__main__':
    import json
    with open('configs/config_celeba.json', 'r') as f:
        config = json.load(f)
    trainer = Trainer(**config)
    with open(trainer.log_dir / 'config.json', 'w') as f:
        json.dump(config, f, indent=4)
    trainer.fit()

