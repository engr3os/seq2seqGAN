import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import cPickle
from dataloader import DataLoader
import h5py
import numpy as np
import os
import tensorflow as tf
import time
from utils import latent_viz_pure, save_h5, latent_viz_mixed
import seq2seqgan
import pdb

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--save_dir',           type=str,   default='./summaries', help='directory to store checkpointed models')
    parser.add_argument('--val_frac',           type=float, default=0.1,        help='fraction of data to be witheld in validation set')
    parser.add_argument('--ckpt_name',          type= str,  default='',         help='name of checkpoint file to load (blank means none)')

    parser.add_argument('--batch_size',         type=int,   default= 64,        help='minibatch size')
    parser.add_argument('--state_dim',          type=int,   default=  51,       help='number of state variables')
    parser.add_argument('--action_dim',         type=int,   default=  2,        help='number of action variables')
    parser.add_argument('--z_dim',              type=int,   default=  128,      help='dimensions of latent variable')
    parser.add_argument('--sample_size',        type=int,   default=  3,        help='number of samples from z')
    parser.add_argument('--sampling_probability',type=int,  default=  0.0,     help='Probability of sampling RNN output')

    parser.add_argument('--num_epochs',         type=int,   default= 500,        help='number of epochs')
    parser.add_argument('--learning_rate',      type=float, default= 0.004,     help='learning rate')
    parser.add_argument('--beta1',              type=float, default= 0.5,       help='Momentum for Adam Update')
    parser.add_argument('--decay_rate',         type=float, default= 0.1,       help='decay rate for learning rate')
    parser.add_argument('--grad_clip',          type=float, default= 5.0,       help='clip gradients at this value')
    parser.add_argument('--save_h5',            type=bool,  default=False,      help='Whether to save network params to h5 file')

    ###############################
    #          Encoder            #
    ###############################
    parser.add_argument('--encoder_size',          type=int,   default=128,        help='number of neurons in each LSTM layer')
    parser.add_argument('--decoder_size',          type=int,   default=128,        help='number of neurons in each LSTM layer')
    parser.add_argument('--num_encoder_layers',    type=int,   default=  2,        help='number of layers in the encoder LSTM')
    parser.add_argument('--num_decoder_layers',    type=int,   default=  2,        help='number of layers in the decoder LSTM')
    parser.add_argument('--seq_len',               type=int,   default=50,         help='LSTM sequence length')

    ############################
    #       Policy Network     #
    ############################
    parser.add_argument('--policy_size',        type=int,   default=128,        help='number of neurons in each feedforward layer')
    parser.add_argument('--num_policy_layers',  type=int,   default=  2,        help='number of layers in the policy network')
    parser.add_argument('--recurrent',          type=bool,  default= False,     help='whether to use recurrent policy')
    parser.add_argument('--dropout_level',      type=float, default=  1.0,      help='percent of state values to keep')


    args = parser.parse_args()

    # Construct model
    net = seq2seqgan.PolicyModel(args)

    # Export model parameters or perform training
    if args.save_h5:
        data_loader = DataLoader(args.batch_size, args.val_frac, args.seq_len*2)
        save_h5(args, net, data_loader)
    else:
        train(args, net)
        
def val_loss(args, data_loader, net, sess):
    data_loader.reset_batchptr_val()
    perfs = [],[],[],[],[],[],[] # perf  = self.flikelihood, self.practions, self.reward, self.gen_loss, self.dis_loss, self.acc_fake, self.acc_real
    for b in xrange(data_loader.n_batches_val):
        # Get batch of inputs/targets
        batch_dict = data_loader.next_batch_val()
        s = batch_dict["states"]
        a = batch_dict["actions"]
        #pdb.set_trace()
        #self.states = [self.enc_state, self.dec_state, self.ddec_state]       
        inputs = [net.pstates, net.pactions, net.fstates, net.factions]
        input_vals = [s[:,:args.seq_len]] + [a[:,:args.seq_len]] + [s[:,args.seq_len:]] + [a[:,args.seq_len:]]                
        """states, samples, enc_output = net.run([net.states, net.z_samples, net.enc_output], sess, inputs, input_vals)
        dec_state = states[-1]
        dec_state_list = list(dec_state)
        last_dec = dec_state_list[-1]
        last_dec = last_dec._replace(h=samples)
        dec_state_list[-1] = last_dec
        dec_state = tuple(dec_state_list)
        states[-1] = dec_state""" 
        [perfs[ind].append(item) for ind, item in enumerate(net.run(net.perfs, sess, inputs, input_vals))]
        
    return [np.array(item) for item in perfs]

# Train network
def train(args, net):
    data_loader = DataLoader(args.batch_size, args.val_frac, args.seq_len*2)
    # Begin tf session
    with tf.Session() as sess:
        #Function to evaluate loss on validation set
        tf.global_variables_initializer().run()
        saver = tf.train.Saver(tf.global_variables(), max_to_keep=5)
        # load from previous save
        if len(args.ckpt_name) > 0:
            print("Restoring old model")
            saver.restore(sess, os.path.join(args.save_dir, args.ckpt_name))

        # Initialize variable to track validation score over time
        old_score = 1e6
        count_decay = 0
        decay_epochs = []

        # Initialize loss
        d_loss, g_loss, r_acc, f_acc = 0.0, 0.0, 0.0, 0.0

        # Set initial learning rate and weight on kl divergence
        print 'setting learning rate to ', args.learning_rate
        sess.run(tf.assign(net.learning_rate, args.learning_rate))

        # Set up tensorboard summary
        merged = tf.summary.merge_all()
        writer = tf.summary.FileWriter('summaries'+'/train', sess.graph)
        test_writer =tf.summary.FileWriter('summaries' + '/test')
        #writer.add_summary(summaries)

        # Loop over epochs
        for e in xrange(args.num_epochs):

            # Evaluate loss on validation set
            test_result = val_loss(args, data_loader, net, sess)
            score = test_result[-4].mean() + test_result[-3].mean()
            print('Validation Loss: {0:f}, '.format(score))
            #test_summary = tf.summary.scalar("Validation_loss", tf.to_float(score)).eval(session=sess)
            print('Real accuracy: {0:f}, '.format(test_result[-1].mean()))
            print('Fake accuracy: {0:f}, '.format(test_result[-2].mean()))
            print('Gen loss: {0:f}, '.format(test_result[-4].mean()))
            print('Disc loss: {0:f}, '.format(test_result[-3].mean()))
            if test_result[-4].mean() < 0.0: pdb.set_trace()
            #test_writer.add_summary(test_summary, e*data_loader.n_batches_train)
            
            # Create plot of latent space
            #latent_viz_pure(args, net, e, sess, data_loader)

            # Set learning rate
            if (old_score - score) < 0.1:
                count_decay += 1
                decay_epochs.append(e)
                if len(decay_epochs) >= 10000 and np.sum(np.diff(decay_epochs)[-2:]) == 2: break
                if args.learning_rate * (args.decay_rate ** count_decay) > 1e-8:
                    print 'setting learning rate to ', args.learning_rate * (args.decay_rate ** count_decay)
                    sess.run(tf.assign(net.learning_rate, args.learning_rate * (args.decay_rate ** count_decay)))
                
            old_score = score

            data_loader.reset_batchptr_train()
            # Loop over batches
            for b in xrange(data_loader.n_batches_train):
                start = time.time()
                # Get batch of inputs/targets
                batch_dict = data_loader.next_batch_train()
                s = batch_dict["states"]
                a = batch_dict["actions"]
                # Set state and action input for encoder
                #states = net.run(net.states,sess)
                inputs = [net.pstates, net.pactions, net.fstates, net.factions]
                input_vals = [s[:,:args.seq_len]] + [a[:,:args.seq_len]] + [s[:,args.seq_len:]] + [a[:,args.seq_len:]]
                _,_ = net.run([net.g_op, net.d_op], sess, inputs, input_vals)
                outputs = [net.g_op, net.gen_loss, net.dis_loss, net.acc_fake, net.acc_real, merged]
                _, g_loss_, d_loss_, f_acc_, r_acc_, summary = net.run(outputs, sess, inputs, input_vals)
                g_loss += g_loss_
                d_loss += d_loss_
                r_acc += r_acc_
                f_acc += f_acc_
                end = time.time()
                writer.add_summary(summary, e*data_loader.n_batches_train)
                # Print loss
                if (e * data_loader.n_batches_train + b) % 10 == 0 and b > 0:
                    print "Training performance: "
                    print "{}/{} (epoch {}), d_loss = {:.3f}, time/batch = {:.3f}" \
                      .format(e * data_loader.n_batches_train + b,
                              args.num_epochs * data_loader.n_batches_train,
                              e, d_loss/10., end - start)
                    print "{}/{} (epoch {}), g_loss = {:.3f}, time/batch = {:.3f}" \
                      .format(e * data_loader.n_batches_train + b,
                              args.num_epochs * data_loader.n_batches_train,
                              e, g_loss/10., end - start)
                    print "{}/{} (epoch {}), r_acc = {:.3f}, time/batch = {:.3f}" \
                      .format(e * data_loader.n_batches_train + b,
                              args.num_epochs * data_loader.n_batches_train,
                              e, r_acc/10., end - start)
                    print "{}/{} (epoch {}), f_acc = {:.3f}, time/batch = {:.3f}" \
                      .format(e * data_loader.n_batches_train + b,
                              args.num_epochs * data_loader.n_batches_train,
                              e, f_acc/10., end - start)
                    d_loss, g_loss, r_acc, f_acc = 0.0, 0.0, 0.0, 0.0

            
            # Save model every epoch
            checkpoint_path = os.path.join(args.save_dir, 'seq2seqgan.ckpt')
            saver.save(sess, checkpoint_path, global_step = e)
            print "model saved to {}".format(checkpoint_path)

if __name__ == '__main__':
    main()
