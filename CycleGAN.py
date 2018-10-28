#
#  █████  ██████  ██████  ██      ███████ ███████ ██████   ██████  ██████   █████  ███    ██  ██████  ███████ ███████
# ██   ██ ██   ██ ██   ██ ██      ██      ██           ██ ██    ██ ██   ██ ██   ██ ████   ██ ██       ██      ██
# ███████ ██████  ██████  ██      █████   ███████  █████  ██    ██ ██████  ███████ ██ ██  ██ ██   ███ █████   ███████
# ██   ██ ██      ██      ██      ██           ██ ██      ██    ██ ██   ██ ██   ██ ██  ██ ██ ██    ██ ██           ██
# ██   ██ ██      ██      ███████ ███████ ███████ ███████  ██████  ██   ██ ██   ██ ██   ████  ██████  ███████ ███████



import numpy as np
from keras.datasets import mnist
from keras.optimizers import Adam
from keras.models import Sequential, Model
from keras.layers import Input, Flatten, Dense, Reshape, Conv2D, UpSampling2D, Concatenate, Dropout, BatchNormalization
from keras.layers.advanced_activations import LeakyReLU
import matplotlib.pyplot as plt
from glob import glob
import scipy
import datetime
import os



class CycleGAN():
    #
    # ██████  ██    ██ ██ ██      ██████       ██████ ██    ██  ██████ ██      ███████  ██████   █████  ███    ██
    # ██   ██ ██    ██ ██ ██      ██   ██     ██       ██  ██  ██      ██      ██      ██       ██   ██ ████   ██
    # ██████  ██    ██ ██ ██      ██   ██     ██        ████   ██      ██      █████   ██   ███ ███████ ██ ██  ██
    # ██   ██ ██    ██ ██ ██      ██   ██     ██         ██    ██      ██      ██      ██    ██ ██   ██ ██  ██ ██
    # ██████   ██████  ██ ███████ ██████       ██████    ██     ██████ ███████ ███████  ██████  ██   ██ ██   ████


    def __init__(self):
        self.img_rows = 128
        self.img_columns = 128
        self.channels = 3
        self.img_shape = (self.img_rows, self.img_columns, self.channels)

        # Calculate output shape of D (PatchGAN)
        patch = int(self.img_rows / 2**4)
        self.disc_patch = (patch, patch, 1)


        #Configure dataset
        self.dataset_name = 'apple2orange'
        self.dataloader = DataLoader(dataset_name=self.dataset_name, img_res=(self.img_rows,self.img_columns))

        optimizer = Adam(lr=0.0002, beta_1=0.5)

        self.lambda_cycle = 10.0
        self.lambda_id = 0.1*self.lambda_cycle


        #Build 2 sets of discriminators

        self.d_A = self.build_discriminator()
        self.d_B = self.build_discriminator()

        #Complie discriminators
        self.d_A.compile(loss='mse', optimizer=optimizer, metrics=['accuracy'])
        self.d_B.compile(loss='mse', optimizer=optimizer, metrics=['accuracy'])

        #Build 2 sets of generators

        self.g_AB = self.build_generator()
        self.g_BA = self.build_generator()

        #Build CycleGAN

        img_A = Input(shape=self.img_shape)
        img_B = Input(shape=self.img_shape)

        #Translate image to other domain
        fake_B = self.g_AB(img_A)
        fake_A = self.g_BA(img_B)

        #reconstruct images back to the original domain
        recon_A = self.g_BA(fake_B)
        recon_B = self.g_AB(fake_A)

        #Identity mapping of images for ID_loss
        img_A_id = self.g_BA(img_A)
        img_B_id = self.g_AB(img_B)

        #set discriminators as not trainable
        self.d_A.trainable = False
        self.d_B.trainable = False

        #pass the fake imgs through discriminators
        valid_A = self.d_A(fake_A)
        valid_B = self.d_B(fake_B)

        #combined model

        self.combined = Model(inputs=[img_A, img_B], outputs=[valid_A, valid_B, recon_A, recon_B, img_A_id, img_B_id])

        self.combined.compile(loss=['mse', 'mse', 'mae', 'mae', 'mae', 'mae'], loss_weights=[1, 1, self.lambda_cycle, self.lambda_cycle, self.lambda_id, self.lambda_id], optimizer=optimizer)

        self.combined.summary()


    def build_generator(self):
        #
        # ██████  ██    ██ ██ ██      ██████           ██████  ███████ ███    ██ ███████ ██████   █████  ████████  ██████  ██████
        # ██   ██ ██    ██ ██ ██      ██   ██         ██       ██      ████   ██ ██      ██   ██ ██   ██    ██    ██    ██ ██   ██
        # ██████  ██    ██ ██ ██      ██   ██         ██   ███ █████   ██ ██  ██ █████   ██████  ███████    ██    ██    ██ ██████
        # ██   ██ ██    ██ ██ ██      ██   ██         ██    ██ ██      ██  ██ ██ ██      ██   ██ ██   ██    ██    ██    ██ ██   ██
        # ██████   ██████  ██ ███████ ██████  ███████  ██████  ███████ ██   ████ ███████ ██   ██ ██   ██    ██     ██████  ██   ██

        #define inner functions
        def conv2d(input_layer, N_filters, filter_size=4):
            #Downsampling
            d = Conv2D(filters=N_filters, kernel_size=filter_size, strides=(2, 2), padding='same')(input_layer)
            d = LeakyReLU(alpha=0.2)(d)
            d = BatchNormalization()(d)
            return d

        def deconv2d(input_layer, skip_input, N_filters, filter_size=4):
            #Upsampling
            u = UpSampling2D(size=(2, 2))(input_layer)
            u = Conv2D(filters=N_filters, kernel_size=filter_size, strides=(1, 1), padding='same', activation='relu')(u)
            u = BatchNormalization()(u)
            u = Concatenate()([u, skip_input])
            return u


        #Build U-Net generator

        #Downsampling
        d0 = Input(shape=self.img_shape)
        d1 = conv2d(d0, 32)
        d2 = conv2d(d1, 64)
        d3 = conv2d(d2, 128)
        d4 = conv2d(d3, 256)

        #Upsampling
        u1 = deconv2d(d4, d3, 128)
        u2 = deconv2d(u1, d2, 64)
        u3 = deconv2d(u2, d1, 32)

        u4 = UpSampling2D(size=(2, 2))(u3)
        output_img = Conv2D(filters=self.channels, kernel_size=4, strides=(1, 1), padding='same', activation='tanh')(u4)

        model = Model(d0, output_img)

        model.summary()

        return model

    def build_discriminator(self):


        #
        # ██████  ██    ██ ██ ██      ██████          ██████  ██ ███████  ██████ ██████  ██ ███    ███ ██ ███    ██  █████  ████████  ██████  ██████
        # ██   ██ ██    ██ ██ ██      ██   ██         ██   ██ ██ ██      ██      ██   ██ ██ ████  ████ ██ ████   ██ ██   ██    ██    ██    ██ ██   ██
        # ██████  ██    ██ ██ ██      ██   ██         ██   ██ ██ ███████ ██      ██████  ██ ██ ████ ██ ██ ██ ██  ██ ███████    ██    ██    ██ ██████
        # ██   ██ ██    ██ ██ ██      ██   ██         ██   ██ ██      ██ ██      ██   ██ ██ ██  ██  ██ ██ ██  ██ ██ ██   ██    ██    ██    ██ ██   ██
        # ██████   ██████  ██ ███████ ██████  ███████ ██████  ██ ███████  ██████ ██   ██ ██ ██      ██ ██ ██   ████ ██   ██    ██     ██████  ██   ██




        #Define inner funnction
        def d_layer(input_layer, N_filters, filter_size=4, normalization=True):

            l = Conv2D(filters=N_filters, kernel_size=filter_size, strides=(2, 2), padding='same')(input_layer)
            l = LeakyReLU(alpha=0.2)(l)
            if normalization:
                l = BatchNormalization()(l)
            return l

        #patchGAN discriminator

        img = Input(shape=self.img_shape)

        l1 = d_layer(img, 64, normalization=False)
        l2 = d_layer(l1, 128)
        l3 = d_layer(l2, 256)
        l4 = d_layer(l3, 512)

        validity = Conv2D(filters=1, kernel_size=4, strides=(1, 1), padding='same')(l4)

        model = Model(img, validity)

        model.summary()

        return model



    def train(self, epochs, batch_size, sampling_interval=50, saving_interval=50):

        #
        # ████████ ██████   █████  ██ ███    ██
        #    ██    ██   ██ ██   ██ ██ ████   ██
        #    ██    ██████  ███████ ██ ██ ██  ██
        #    ██    ██   ██ ██   ██ ██ ██  ██ ██
        #    ██    ██   ██ ██   ██ ██ ██   ████


        start_time = datetime.datetime.now()

        #labels
        valid = np.ones((batch_size,) + self.disc_patch)
        fake = np.zeros((batch_size,) + self.disc_patch)

        for epoch in range(epochs):
            for batch_i, (imgs_A, imgs_B) in enumerate(self.dataloader.load_batch(batch_size)):

                #
                # ████████ ██████   █████  ██ ███    ██     ██████  ██ ███████
                #    ██    ██   ██ ██   ██ ██ ████   ██     ██   ██ ██ ██
                #    ██    ██████  ███████ ██ ██ ██  ██     ██   ██ ██ ███████
                #    ██    ██   ██ ██   ██ ██ ██  ██ ██     ██   ██ ██      ██
                #    ██    ██   ██ ██   ██ ██ ██   ████     ██████  ██ ███████





                #translate images to opposite domain
                fake_B = self.g_AB.predict(imgs_A)
                fake_A = self.g_BA.predict(imgs_B)

                #train discriminators on real and fake data
                dA_loss_real = self.d_A.train_on_batch(imgs_A, valid)
                dA_loss_fake = self.d_A.train_on_batch(fake_A, fake)
                dA_loss = 0.5 * np.add(dA_loss_real, dA_loss_fake)


                dB_loss_real = self.d_B.train_on_batch(imgs_B, valid)
                dB_loss_fake = self.d_B.train_on_batch(fake_B, fake)
                dB_loss = 0.5 * np.add(dB_loss_real, dB_loss_fake)

                #total discriminator loss
                d_loss = 0.5 * np.add(dA_loss, dB_loss)


                #
                # ████████ ██████   █████  ██ ███    ██      ██████  ███████ ███    ██
                #    ██    ██   ██ ██   ██ ██ ████   ██     ██       ██      ████   ██
                #    ██    ██████  ███████ ██ ██ ██  ██     ██   ███ █████   ██ ██  ██
                #    ██    ██   ██ ██   ██ ██ ██  ██ ██     ██    ██ ██      ██  ██ ██
                #    ██    ██   ██ ██   ██ ██ ██   ████      ██████  ███████ ██   ████




                g_loss = self.combined.train_on_batch([imgs_A, imgs_B], [valid, valid, imgs_A, imgs_B, imgs_A, imgs_B])

                elapsed_time = datetime.datetime.now() - start_time

                #plot the progress
                print("[Epoch: %d/%d] [Batch: %d/%d] [D loss: %f, acc: %3d%%] [G loss: %05f, adv: %05f, recon: %05f, id: %05f] time: %s " % (epoch, epochs, batch_i, self.dataloader.n_batches, d_loss[0], 100*d_loss[1], g_loss[0], np.mean(g_loss[1:3]), np.mean(g_loss[3:5]), np.mean(g_loss[5:6]), elapsed_time))


                if (batch_i % sampling_interval == 0) or (batch_i==self.dataloader.n_batches-1):
                    self.sample_images(epoch, batch_i)

                if (batch_i % saving_interval == 0) or (batch_i==self.dataloader.n_batches-1):
                    self.saver()





    def saver(self):

        #
        # ███████  █████  ██    ██ ███████ ██████
        # ██      ██   ██ ██    ██ ██      ██   ██
        # ███████ ███████ ██    ██ █████   ██████
        #      ██ ██   ██  ██  ██  ██      ██   ██
        # ███████ ██   ██   ████   ███████ ██   ██

        os.makedirs('saved_models/%s' % self.dataset_name, exist_ok=True)

        #saving generator to disk
        G_AB_json = self.g_AB.to_json()
        with open("saved_models/%s/G_AB.json" % self.dataset_name, "w") as json_file:
            json_file.write(G_AB_json)
        # serialize weights to HDF5
        self.g_AB.save_weights("saved_models/%s/G_AB.h5" % self.dataset_name)

        G_BA_json = self.g_BA.to_json()
        with open("saved_models/%s/G_BA.json" % self.dataset_name, "w") as json_file:
            json_file.write(G_BA_json)
        # serialize weights to HDF5
        self.g_BA.save_weights("saved_models/%s/G_BA.h5" % self.dataset_name)
        print("Saved Gs to disk")


        #saving discriminator to disk
        D_A_json = self.d_A.to_json()
        with open("saved_models/%s/D_A.json" % self.dataset_name, "w") as json_file:
            json_file.write(D_A_json)
        # serialize weights to HDF5
        self.d_A.save_weights("saved_models/%s/D_A.h5" % self.dataset_name)

        D_B_json = self.d_B.to_json()
        with open("saved_models/%s/D_B.json" % self.dataset_name, "w") as json_file:
            json_file.write(D_B_json)
        # serialize weights to HDF5
        self.d_B.save_weights("saved_models/%s/D_B.h5" % self.dataset_name)

        print("Saved Ds to disk")






    def sample_images(self, epoch, batch_i):
        #
        # ███████  █████  ███    ███ ██████  ██      ███████         ██ ███    ███  █████   ██████  ███████ ███████
        # ██      ██   ██ ████  ████ ██   ██ ██      ██              ██ ████  ████ ██   ██ ██       ██      ██
        # ███████ ███████ ██ ████ ██ ██████  ██      █████           ██ ██ ████ ██ ███████ ██   ███ █████   ███████
        #      ██ ██   ██ ██  ██  ██ ██      ██      ██              ██ ██  ██  ██ ██   ██ ██    ██ ██           ██
        # ███████ ██   ██ ██      ██ ██      ███████ ███████ ███████ ██ ██      ██ ██   ██  ██████  ███████ ███████

        os.makedirs('images/%s' % self.dataset_name, exist_ok=True)
        r, c=2, 3

        imgs_A = self.dataloader.load_data(domain="A", batch_size=1, is_testing=True)
        imgs_B = self.dataloader.load_data(domain="B", batch_size=1, is_testing=True)

        fake_B = self.g_AB.predict(imgs_A)
        fake_A = self.g_BA.predict(imgs_B)

        #reconstruct images back to the original domain
        recon_A = self.g_BA.predict(fake_B)
        recon_B = self.g_AB.predict(fake_A)

        gen_imgs = np.concatenate([imgs_A, fake_B, recon_A, imgs_B, fake_A, recon_B])

        #rescale images
        gen_imgs = 0.5*gen_imgs + 0.5

        titles = ['Original', 'Translated', 'Reconstructed']
        fig, axs = plt.subplots(r, c)
        cnt = 0
        for i in range(r):
            for j in range(c):
                axs[i,j].imshow(gen_imgs[cnt])
                axs[i, j].set_title(titles[j])
                axs[i,j].axis('off')
                cnt += 1
        fig.savefig("images/%s/%d_%d.png" % (self.dataset_name, epoch, batch_i))
        plt.close()



class DataLoader():

    #
    # ██████   █████  ████████  █████  ██       ██████   █████  ██████  ███████ ██████
    # ██   ██ ██   ██    ██    ██   ██ ██      ██    ██ ██   ██ ██   ██ ██      ██   ██
    # ██   ██ ███████    ██    ███████ ██      ██    ██ ███████ ██   ██ █████   ██████
    # ██   ██ ██   ██    ██    ██   ██ ██      ██    ██ ██   ██ ██   ██ ██      ██   ██
    # ██████  ██   ██    ██    ██   ██ ███████  ██████  ██   ██ ██████  ███████ ██   ██

    def __init__(self, dataset_name, img_res=(128,128)):
        self.dataset_name = dataset_name
        self.img_res = img_res

    def load_data(self, domain, batch_size=1, is_testing=False):


        #
        # ██       ██████   █████  ██████  ███████ ██████
        # ██      ██    ██ ██   ██ ██   ██ ██      ██   ██
        # ██      ██    ██ ███████ ██   ██ █████   ██████
        # ██      ██    ██ ██   ██ ██   ██ ██      ██   ██
        # ███████  ██████  ██   ██ ██████  ███████ ██   ██



        data_type = "train%s" % domain if not is_testing else "test%s" % domain
        path = glob('./datasets/%s/%s/*' % (self.dataset_name, data_type))

        batch_images = np.random.choice(path, size=batch_size)

        imgs = []
        for img_path in batch_images:
            img = self.imread(img_path)
            if not is_testing:
                img = scipy.misc.imresize(img, self.img_res)

                if np.random.random()>0.5:
                    img = np.fliplr(img)
            else:
                img = scipy.misc.imresize(img, self.img_res)
            imgs.append(img)

        imgs = np.array(imgs)/127.5 - 1

        return imgs


    def load_batch(self, batch_size=1, is_testing=False):


        #
        # ██████   █████  ████████  █████           ██████  ███████ ███    ██ ███████ ██████   █████  ████████  ██████  ██████
        # ██   ██ ██   ██    ██    ██   ██         ██       ██      ████   ██ ██      ██   ██ ██   ██    ██    ██    ██ ██   ██
        # ██   ██ ███████    ██    ███████         ██   ███ █████   ██ ██  ██ █████   ██████  ███████    ██    ██    ██ ██████
        # ██   ██ ██   ██    ██    ██   ██         ██    ██ ██      ██  ██ ██ ██      ██   ██ ██   ██    ██    ██    ██ ██   ██
        # ██████  ██   ██    ██    ██   ██ ███████  ██████  ███████ ██   ████ ███████ ██   ██ ██   ██    ██     ██████  ██   ██


        data_type = "train" if not is_testing else "val"
        path_A = glob('./datasets/%s/%sA/*' % (self.dataset_name, data_type))
        path_B = glob('./datasets/%s/%sB/*' % (self.dataset_name, data_type))

        self.n_batches = int(min(len(path_A),len(path_B))/batch_size)
        total_samples = self.n_batches*batch_size

        path_A = np.random.choice(path_A, total_samples, replace=False)
        path_B = np.random.choice(path_B, total_samples, replace=False)

        print(path_A.shape)
        print(self.n_batches)

        for i in range(self.n_batches-1):
            batch_A = path_A[i*batch_size:(i+1)*batch_size]
            batch_B = path_B[i*batch_size:(i+1)*batch_size]
            imgs_A, imgs_B = [], []
            for img_A, img_B in zip(batch_A, batch_B):
                img_A = self.imread(img_A)
                img_B = self.imread(img_B)

                img_A = scipy.misc.imresize(img_A, self.img_res)
                img_B = scipy.misc.imresize(img_B, self.img_res)

                if not is_testing and np.random.random()>0.5:
                    img_A = np.fliplr(img_A)
                    img_B = np.fliplr(img_B)

                imgs_A.append(img_A)
                imgs_B.append(img_B)

            imgs_A = np.array(imgs_A)/127.5 - 1.
            imgs_B = np.array(imgs_B)/127.5 - 1.

            yield imgs_A, imgs_B


    def load_img(self, path):
        img = self.imread(path)
        img = scipy.misc.imresize(img, self.img_res)
        img = img/127.5 - 1.
        return img[np.newaxis, :, :, :]

    def imread(self,path):
        return scipy.misc.imread(path, mode='RGB').astype(np.float)




def main():

    #
    # ███    ███  █████  ██ ███    ██     ███    ███ ███████ ████████ ██   ██  ██████  ██████
    # ████  ████ ██   ██ ██ ████   ██     ████  ████ ██         ██    ██   ██ ██    ██ ██   ██
    # ██ ████ ██ ███████ ██ ██ ██  ██     ██ ████ ██ █████      ██    ███████ ██    ██ ██   ██
    # ██  ██  ██ ██   ██ ██ ██  ██ ██     ██  ██  ██ ██         ██    ██   ██ ██    ██ ██   ██
    # ██      ██ ██   ██ ██ ██   ████     ██      ██ ███████    ██    ██   ██  ██████  ██████


    gan = CycleGAN()
    gan.train(epochs=200, batch_size=1, sampling_interval=50, saving_interval=50)

if __name__ == '__main__':
    main()
