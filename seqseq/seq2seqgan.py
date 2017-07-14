import numpy as np
import tensorflow as tf
from tensorflow.contrib import rnn
from tensorflow.contrib import seq2seq
from tensorflow.python.layers.core import Dense
from tensorflow.contrib.layers.python.layers import initializers
from sampler_fns import initialize_fn
from sampler_fns import sample_fn
from sampler_fns import next_inputs_fn
from decoder_new import dynamic_decode
from helper_new import CustomHelper
from basic_decoder_new import BasicDecoder
import pdb
import cPickle
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

class PolicyModel():
    def __init__(self, args):
        # Placeholder for data
        self.pstates = tf.placeholder(tf.float32, [args.batch_size, args.seq_len, args.state_dim], name="past_states")
        self.pactions = tf.placeholder(tf.float32, [args.batch_size, args.seq_len, args.action_dim], name="past_actions")
        self.fstates = tf.placeholder(tf.float32, [args.batch_size, args.seq_len, args.state_dim], name="future_states")
        self.factions = tf.placeholder(tf.float32, [args.batch_size, args.seq_len, args.action_dim], name="future_actions")
        self.learning_rate = tf.Variable(0.0, trainable=False, name="learning_rate")

        # Create the computational graph
        self.enc_lstm_units = [rnn.LSTMCell(args.encoder_size, state_is_tuple=True, initializer=initializers.xavier_initializer()) for _ in range(args.num_encoder_layers)]
        #self.enc_lstm_units.append(rnn.LSTMCell(args.encoder_size, num_proj=args.action_dim, state_is_tuple=True, initializer=initializers.xavier_initializer()))
        self.enc_lstm_cell = rnn.MultiRNNCell(self.enc_lstm_units, state_is_tuple=True)
        self.enc_state = self.enc_lstm_cell.zero_state(args.batch_size*args.sample_size, tf.float32)
        self.dec_lstm_units = [rnn.LSTMCell(args.decoder_size, state_is_tuple=True, initializer=initializers.xavier_initializer()) for _ in range(args.num_decoder_layers)]
        #self.dec_lstm_units.append(rnn.LSTMCell(args.decoder_size, num_proj=args.action_dim, state_is_tuple=True, initializer=initializers.xavier_initializer()))
        
        self.dec_lstm_cell = rnn.MultiRNNCell(self.dec_lstm_units, state_is_tuple=True)
        #self.dec_state = self.dec_lstm_cell.zero_state(args.batch_size*args.sample_size, tf.float32)
        self.ddec_lstm_units = [rnn.LSTMCell(args.decoder_size, state_is_tuple=True, initializer=initializers.xavier_initializer()) for _ in range(args.num_decoder_layers)]
        self.ddec_lstm_cell = rnn.MultiRNNCell(self.ddec_lstm_units, state_is_tuple=True)
        #self.ddec_state = self.ddec_lstm_cell.zero_state(args.batch_size*args.sample_size, tf.float32)
        
        self.optimizer(args)
        self.perfs = self.flikelihood, self.practions, self.reward, self.gen_loss, self.dis_loss, self.acc_fake, self.acc_real

   
    def encoder(self, pstates, pactions, args):    
        # Create the encoder
        encoder_input = tf.concat([pstates, pactions], 2)
        encoder_input = tf.reverse(encoder_input, [1])
        encoder_input = self.add_sample_axis(encoder_input, args)
        enc_helper = seq2seq.TrainingHelper(encoder_input, sequence_length=[args.seq_len]*(args.batch_size*args.sample_size))
        encoder = seq2seq.BasicDecoder(cell=self.enc_lstm_cell, helper=enc_helper, initial_state=self.enc_state)
        enc_output, self.enc_state, _ = seq2seq.dynamic_decode(encoder)
        istates = self.enc_state
        return istates
        
    def zero_input(self, args):
        # Fully connected layer to latent variable distribution parameters
        W_samp = tf.get_variable("samp_w", [args.encoder_size, args.action_dim], initializer=initializers.xavier_initializer())
        b_samp = tf.get_variable("samp_b", [args.action_dim], initializer=tf.constant_initializer(0.0))
        zero_inp = tf.nn.xw_plus_b(self.enc_state[-1].h, W_samp, b_samp)  # seq2seq ouput contains tensor[0] and id[1]
        # Create sampler
        return tf.concat([tf.expand_dims(zero_inp,1), self.add_sample_axis(self.factions, args)], 1)
        
    def generator(self, args):
        # Create decoder
        #pdb.set_trace() 
        out_layer = Dense(args.action_dim*2)
        dec_input = self.zero_input(args)
        dec_helper = CustomHelper(initialize_fn, sample_fn, next_inputs_fn, inputs=dec_input, 
                                          sequence_length=[args.seq_len]*(args.batch_size*args.sample_size), 
                                            sampling_probability=args.sampling_probability, seed=None)
        #dec_helper = seq2seq.ScheduledOutputTrainingHelper(dec_input, sequence_length=[args.seq_len]*(args.batch_size*args.sample_size), sampling_probability=1.0)
        decoder = BasicDecoder(cell=self.dec_lstm_cell, helper=dec_helper, initial_state=self.dec_state, output_layer=out_layer)
        dec_output, dec_final_input, self.dec_state = dynamic_decode(decoder, maximum_iterations=args.seq_len)
        dec_output, sample_ids = dec_output
        #self.sample_ids = sample_ids
        z_mean, z_logstd = tf.split(dec_output, 2, 2)
        z_logstd = tf.nn.elu(z_logstd)
        samples = (dec_final_input-z_mean)/tf.exp(z_logstd)
        z_nll =  0.5*tf.reduce_sum(tf.square(samples),2) + tf.reduce_sum(z_logstd,2) + 0.5*tf.to_float(args.action_dim) * np.log(2.0*np.pi)   
        z_nll = tf.reduce_sum(z_nll,1)
        f_action = dec_final_input #tf.reshape(dec_output, [args.batch_size*args.sample_size, args.seq_len, args.action_dim])
        self.dec_input, self.dec_output, self.sample_ids, self.z_nll, self.pr_action = dec_input, dec_output, sample_ids, z_nll, f_action
        return  z_nll, f_action, sample_ids    # Initialize logstd
            
    def discriminator(self, f_actions, args):
        ddec_helper = seq2seq.TrainingHelper(f_actions, sequence_length=[args.seq_len]*(args.batch_size*args.sample_size))
        ddecoder = seq2seq.BasicDecoder(cell=self.ddec_lstm_cell, helper=ddec_helper, initial_state=self.ddec_state)
        ddec_output, self.ddec_state, _ = seq2seq.dynamic_decode(ddecoder)
        # Fully connected layer to latent variable distribution parameters
        W_ddec = tf.get_variable("ddec_w", [args.decoder_size, 1], initializer=initializers.xavier_initializer())
        b_ddec = tf.get_variable("ddec_b", [1])
        return tf.nn.xw_plus_b(self.ddec_state[-1].h, W_ddec, b_ddec)
                
    def optimizer(self, args):
        with tf.variable_scope('Generator'):
            with tf.variable_scope('Encoder'):
                self.dec_state = self.encoder(self.pstates, self.pactions, args)
            nll, pred_factions, sample_ids = self.generator(args)
        flikelihood = tf.exp(-nll)
        sampled = tf.cast(tf.reduce_any(sample_ids,1),tf.float32)
        #self.sampled = sampled
        with tf.variable_scope('Encoder', reuse=True):
            self.ddec_state = self.encoder(self.pstates, self.pactions, args)
        with tf.variable_scope('Discriminator'):
            fake_logits = self.discriminator(pred_factions, args)
        a_factions = self.add_sample_axis(self.factions, args)
        print 'act_future_traj: ', a_factions.get_shape().as_list()
        with tf.variable_scope('Discriminator', reuse=True):
            real_logits = self.discriminator(a_factions, args)
        dis_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=real_logits, labels=tf.ones_like(real_logits))) + \
        tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=fake_logits, labels=tf.zeros_like(fake_logits)))
        reward = tf.sigmoid(fake_logits)
        #nlr = -tf.log(reward)
        nlr = tf.nn.sigmoid_cross_entropy_with_logits(logits=fake_logits, labels=tf.ones_like(fake_logits))
        self.acc_fake = tf.reduce_mean(tf.cast(tf.equal(tf.rint(tf.sigmoid(fake_logits)), tf.zeros_like(fake_logits)), tf.float32))
        self.acc_real = tf.reduce_mean(tf.cast(tf.equal(tf.rint(tf.sigmoid(real_logits)), tf.ones_like(real_logits)), tf.float32))
        #ll, lr = tf.reshape(-nll, [-1, args.sample_size]), tf.reshape(-nlr, [-1, args.sample_size])       
        #gen_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=ll, labels=tf.nn.softmax(lr)))
        #pdb.set_trace()
        gen_loss = tf.reduce_mean((nlr+nll)*sampled + nll*(1-sampled))
        #gen_loss = tf.reduce_mean(nll)
        gen_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope="Generator")
        dis_vars =tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope="Discriminator")
        gen_loss += tf.reduce_sum([tf.reduce_sum(tf.square(x)) for x in gen_vars])
        dis_loss += tf.reduce_sum([tf.reduce_sum(tf.square(x)) for x in dis_vars])
        gen_grads, _ = tf.clip_by_global_norm(tf.gradients(gen_loss, gen_vars), args.grad_clip)
        dis_grads, _ = tf.clip_by_global_norm(tf.gradients(dis_loss, dis_vars), args.grad_clip)
        g_optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate) # , beta1 = args.beta1Adam Optimizer
        d_optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate*1e-1) # Adam Optimizer
        self.g_op = g_optimizer.apply_gradients(zip(gen_grads, gen_vars))
        self.d_op = d_optimizer.apply_gradients(zip(dis_grads, dis_vars))
        self.var_summary(gen_grads, gen_vars)
        self.var_summary(dis_grads, dis_vars)
        tf.summary.scalar("gen_loss", gen_loss)
        tf.summary.scalar("dis_loss", dis_loss)
        tf.summary.scalar("mean_flikelihood", tf.reduce_mean(flikelihood))
        tf.summary.scalar("mean_reward", tf.reduce_mean(reward))
        tf.summary.scalar("acc_fake", self.acc_fake)
        tf.summary.scalar("acc_real", self.acc_real)
        self.params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)
        self.flikelihood, self.practions = tf.reshape(flikelihood, [-1,args.sample_size]), tf.reshape(pred_factions, [-1,args.sample_size,args.seq_len,args.action_dim])
        self.gen_loss, self.dis_loss, self.reward = gen_loss, dis_loss, tf.reshape(reward, [-1,args.sample_size])
        # Plot trajectories
        """
        sl=3
        act_traj = tf.expand_dims(tf.concat([self.pactions, self.factions], 1),1)
        print 'act_traj: ', act_traj.get_shape().as_list()
        act_traj_img = tf.py_func(self.plot_gt, [tf.slice(act_traj, [0,0,0,0],[sl,-1,-1,-1])], tf.float32)
        a_pactions = tf.reshape(self.add_sample_axis(self.pactions, args), [-1,args.sample_size,args.seq_len,args.action_dim])
        pred_traj = tf.concat([a_pactions, self.practions], 2)
        print 'pred_traj: ', pred_traj.get_shape().as_list()
        inputs = [tf.slice(pred_traj,[0,0,0,0],[sl,-1,-1,-1]), tf.slice(self.flikelihood,[0,0],[sl,-1]), tf.slice(self.reward,[0,0],[sl,-1]), args.seq_len]
        pred_traj_img = tf.py_func(self.plot_pred, inputs, tf.float32)
        tf.summary.image('gt_traj', act_traj_img)
        tf.summary.image('pred_trajs', pred_traj_img)
        """
        
        # Emit summaries
    def var_summary(self, grads, variables):
        for grad, var in zip(grads, variables):
            tf.summary.histogram(var.name, var)
            if grad is not None: tf.summary.histogram(var.name + '/gradients', grad)
            
    def run(self, outputs, sess=None, inputs=[], input_vals=[]):
            if not sess: sess = tf.get_default_session()
            return sess.run(outputs, feed_dict=dict(list(zip(inputs, input_vals))))
    
    def save_model(self, filename, sess=None):
        params = self.run(self.params, sess=sess)
        cPickle.dump(params, open(filename, 'w'))
        
    def add_sample_axis(self, tensor, args):
        tensor = tf.tile(tf.expand_dims(tensor,1),[1,args.sample_size,1,1])
        return tf.reshape(tensor, [args.batch_size*args.sample_size,args.seq_len,-1])
        
        
    def plot_gt(self, actions):
        disp_imgs = []
        fig, ax = plt.subplots(figsize=(12.8, 7.2), dpi=100)
        #ax.axis('off')
        #ax.tick_params(axis='both', left='off', top='off', right='off', bottom='off', labelleft='off', labeltop='off', labelright='off', labelbottom='off')
        plt.setp([ax.get_xticklines() + ax.get_yticklines() + ax.get_xgridlines() + ax.get_ygridlines()],antialiased=False)
        for action in actions:
            ax.plot(np.cumsum(action[:,:,0].T,0), np.cumsum(action[:,:,1].T,0))
            fig.canvas.draw()
            ncols, nrows = fig.canvas.get_width_height()
            disp_img = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8).reshape(nrows, ncols, 3)
            ax.cla()
            disp_imgs.append(disp_img)
        plt.close(fig)
        return np.array(disp_imgs, dtype=np.float32)
        
    def plot_pred(self, actions, lls, rewards, seq_len):
        disp_imgs = []
        fig, ax = plt.subplots(figsize=(12.8, 7.2), dpi=100)
        #ax.axis('off')
        #ax.tick_params(axis='both', left='off', top='off', right='off', bottom='off', labelleft='off', labeltop='off', labelright='off', labelbottom='off')
        plt.setp([ax.get_xticklines() + ax.get_yticklines() + ax.get_xgridlines() + ax.get_ygridlines()],antialiased=False)
        ll_id = -seq_len/2
        reward_id = -1        
        for action, ll, reward in zip (actions, lls, rewards):
            #ax.set_xticks([])
        	 #ax.set_yticks([])
            ax.plot(np.cumsum(action[:,:,0].T, 0), np.cumsum(action[:,:,1].T, 0))
            for ll_, reward_, action_ in zip(ll, reward, action):
                ax.annotate(ll_, (action_[ll_id, 0], action_[ll_id, 1]))
                ax.annotate(reward_, (action_[reward_id, 0], action_[reward_id, 1]))
            fig.canvas.draw()
            ncols, nrows = fig.canvas.get_width_height()
            disp_img = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8).reshape(nrows, ncols, 3)
            ax.cla()
            disp_imgs.append(disp_img)
        plt.close(fig)
        return np.array(disp_imgs, dtype=np.float32)





