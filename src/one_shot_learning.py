from utils import OmniglotDataLoader, one_hot_decode, five_hot_decode
from util import eprint, eprint2, str2bool, serialize_plot, gen_exp_name, mkdir
from input_loader import InputLoader
import tensorflow as tf
import argparse
import numpy as np
from model import NTMOneShotLearningModel
from tensorflow.python import debug as tf_debug
from pdb import set_trace
from datetime import datetime

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', default="train")
    parser.add_argument('--restore_training', default=False, type=str2bool)
    parser.add_argument('--summary_writer', default=False, type=str2bool)
    parser.add_argument('--serialize', default=True, type=str2bool)
    parser.add_argument('--model_saver', default=False, type=str2bool)
    parser.add_argument('--use_subset_classes', default=True, type=str2bool)
    parser.add_argument('--use_pretrained', default=False, type=str2bool)
    parser.add_argument('--class_difficulty', default='easy')
    parser.add_argument('--debug', default=False, type=str2bool)
    parser.add_argument('--tf_debug_flag', default=False, type=str2bool)
    parser.add_argument('--label_type', default="one_hot", help='one_hot or five_hot')
    parser.add_argument('--n_classes', default=5, type=int)
    parser.add_argument('--seq_length', default=35, type=int) # Bruh.. Don't use above 35
    parser.add_argument('--sampling_strategy', default='random')
    parser.add_argument('--augment', default=True, type=str2bool)
    parser.add_argument('--im_normalization', default=True, type=str2bool)
    parser.add_argument('--model', default="MANN", help='LSTM, MANN, MANN2 or NTM')
    parser.add_argument('--read_head_num', default=4, type=int)
    parser.add_argument('--batch_size', default=16, type=int)
    parser.add_argument('--num_epoches', default=1000, type=int)
    parser.add_argument('--batches_validation', default=5, type=int)
    parser.add_argument('--model_save_freq', default=500, type=int)
    parser.add_argument('--validation_freq', default=25, type=int)
    parser.add_argument('--learning_rate', default=1e-3, type=int)
    parser.add_argument('--rnn_size', default=200, type=int)
    parser.add_argument('--image_width', default=128, type=int)
    parser.add_argument('--image_height', default=128, type=int)
    parser.add_argument('--rnn_num_layers', default=1, type=int)
    parser.add_argument('--memory_size', default=128, type=int)
    parser.add_argument('--memory_vector_dim', default=40, type=int)
    parser.add_argument('--shift_range', default=1, help='Only for model=NTM')
    parser.add_argument('--write_head_num', default=1, help='Only for model=NTM. For MANN #(write_head) = #(read_head)')
    parser.add_argument('--test_batch_num', default=100, type=int)
    parser.add_argument('--n_train_classes', default=1200, type=int)
    parser.add_argument('--n_test_classes', default=423, type=int)
    parser.add_argument('--save_dir', default='job_outputs/experiments')
    parser.add_argument('--data_dir', default='../../images_background')
    parser.add_argument('--dataset_type', default='kinetics_dynamic') # options: omniglot, kinetics_dynamic, kinetics_video, kinetics_single_frame
    parser.add_argument('--controller_type', default='alex') # options: alex, vgg19, i3d, default 
    parser.add_argument('--sample_nframes', default=64, type=int)
    parser.add_argument('--optimizer', default='adam')
    parser.add_argument('--tensorboard_dir', default='./summary/one_shot_learning')
    args = parser.parse_args()
    np.random.seed(0)
    if args.mode == 'train':
        train(args)
    elif args.mode == 'test':
        test(args)


def train(args):
    eprint("Args: ", args)
    exp_name = gen_exp_name(args)
    eprint(exp_name)
    eprint("Loading in Model")
    model = NTMOneShotLearningModel(args)
    eprint("Loading Data")
    if args.dataset_type == 'omniglot':
        data_loader = OmniglotDataLoader(
            image_size=(args.image_width, args.image_height),
            n_train_classes=args.n_train_classes,
            n_test_classes=args.n_test_classes,
            data_dir=args.data_dir
        )
        test_data_loader = data_loader
    elif args.dataset_type == 'kinetics_dynamic':
        data_loader = InputLoader('dynamic_image', 'train', im_size=args.image_height, args=args)
        test_data_loader = InputLoader('dynamic_image', 'val', im_size=args.image_height, args=args)
    elif args.dataset_type == 'kinetics_video':
        data_loader = InputLoader('raw_video', 'train', args=args, im_size=args.image_height)
        test_data_loader = InputLoader('raw_video', 'val', args=args, im_size=args.image_height)
    elif args.dataset_type == 'kinetics_single_frame':
        data_loader = InputLoader('single_frame', 'train', args=args, im_size=args.image_height)
        test_data_loader = InputLoader('single_frame', 'val', args=args, im_size=args.image_height)


    eprint("Starting Session")
    with tf.Session() as sess:
        eprint("Started Session")
        if args.tf_debug_flag:
            sess = tf_debug.LocalCLIDebugWrapperSession(sess)
        if args.restore_training:
            saver = tf.train.Saver()
            ckpt = tf.train.get_checkpoint_state(args.save_dir + '/' + args.model)
            saver.restore(sess, ckpt.model_checkpoint_path)
        else:
            if args.model_saver:
                eprint("Starting saver")
                saver = tf.train.Saver(tf.global_variables())
                eprint("Finished saver")
            tf.global_variables_initializer().run()
            eprint("Finished Initialization")
        if args.summary_writer:
            train_writer = tf.summary.FileWriter(args.tensorboard_dir + '/' + args.model, sess.graph)
            eprint("Train Writer Finished")
        eprint(args)
        eprint("1st\t2nd\t3rd\t4th\t5th\t6th\t7th\t8th\t9th\t10th\tbatch\tloss")

        loss_list = []
        accuracy_list = []
        for b in range(args.num_epoches):

            # Test

            if b % args.validation_freq == 0:
                learning_loss_list, output_list, y_list = [], [], []
                for val_index in range(args.batches_validation):
                    if args.dataset_type == 'omniglot':
                        x_image, x_label, y = data_loader.fetch_batch(args.n_classes, args.batch_size, args.seq_length,
                                                                type='test',
                                                                sampling_strategy=args.sampling_strategy,
                                                                augment=args.augment,
                                                                label_type=args.label_type)
                    else: 
                        # all kinetics loaders
                        if args.debug:
                            eprint("Loading in validation data")
                        x_image, x_label, y = test_data_loader.fetch_batch(args.n_classes, args.batch_size, args.seq_length,
                                                sampling_strategy=args.sampling_strategy,
                                                augment=args.augment,
                                                label_type=args.label_type)
                    feed_dict = {model.x_image: x_image, model.x_label: x_label, model.y: y, model.is_training: False}
                    if args.debug:
                        eprint("Validation running session")

                    output, learning_loss = sess.run([model.o, model.learning_loss], feed_dict=feed_dict)
                    if args.summary_writer:
                        merged_summary = sess.run(model.learning_loss_summary, feed_dict=feed_dict)
                        train_writer.add_summary(merged_summary, b)
                    learning_loss_list.append(learning_loss)
                    output_list.append(output)
                    y_list.append(y)
                # state_list = sess.run(model.state_list, feed_dict=feed_dict)  # For debugging
                # with open('state_long.txt', 'w') as f:
                #     print(state_list, file=f)
                output_total = np.concatenate(output_list, axis=0)
                y_total = np.concatenate(y_list, axis=0)
                learning_loss = np.mean(learning_loss_list)
                accuracy, total = test_f(args, y_total, output_total)
                eprint(end='\t')
                for accu in accuracy:
                    eprint2('%.4f' % accu, end='\t')
                eprint2(end='\n')
                eprint(end='\t')
                for tot in total:
                    eprint2('%d' % tot, end='\t')

                eprint2('%d\t%.4f' % (b, learning_loss))

                if args.serialize:
                    accuracy_list.append(accuracy)
                    dir_name = args.save_dir + '/' + exp_name
                    mkdir(dir_name)
                    eprint("Serializing intermediate accuracy")
                    serialize_plot(accuracy_list, dir_name, "inter_accuracy")

            # Save model

            if args.model_saver and b % args.model_save_freq == 0 and b > 0:
                if args.debug:
                    eprint("saving to: {}".format(args.save_dir + '/' + args.model + '/model.tfmodel'))
                
                mkdir(args.save_dir + '/' + args.model)
                saver.save(sess, args.save_dir + '/' + args.model + '/model.tfmodel', global_step=b)

            # Train
            if args.debug:
                eprint("[{}] Fetch Batch".format(b))
            if args.dataset_type == 'omniglot':
                x_image, x_label, y = data_loader.fetch_batch(args.n_classes, args.batch_size, args.seq_length,
                                                              type='train',
                                                              sampling_strategy=args.sampling_strategy,
                                                              augment=args.augment,
                                                              label_type=args.label_type)
            else: 
                x_image, x_label, y = data_loader.fetch_batch(args.n_classes, args.batch_size, args.seq_length,
                                                              sampling_strategy=args.sampling_strategy,
                                                              augment=args.augment,
                                                              label_type=args.label_type)

            if args.debug:
                eprint("[{}] Run Sess".format(b))
            #  feed_dict = {model.x_image: x_image, model.x_label: x_label, model.y: y}
            feed_dict = {model.x_image: x_image, model.x_label: x_label, model.y: y, model.is_training: False}
            learning_loss, _ = sess.run([model.learning_loss, model.train_op], feed_dict=feed_dict)
            if args.debug:
                eprint("[{}] Learning Loss: {:.3f}".format(b, learning_loss))
            if args.serialize:
                loss_list.append(learning_loss)
                dir_name = args.save_dir + '/' + exp_name
                mkdir(dir_name)
                rand_val = str(np.random.randint(0, 100))
                eprint("Serializing intermediate loss")
                serialize_plot(loss_list, dir_name, "inter_loss")

    if args.serialize:
        dir_name = args.save_dir + '/' + exp_name
        mkdir(dir_name)
        rand_val = str(np.random.randint(0, 100))
        eprint("Serializing: ", rand_val)
        serialize_plot(loss_list, dir_name, "loss" + rand_val)
        serialize_plot(accuracy_list, dir_name, "accuracy" + rand_val)
        serialize_plot(args, dir_name, "arguments" + rand_val)
        eprint("Saved all plots")



def test(args):
    model = NTMOneShotLearningModel(args)
    data_loader = OmniglotDataLoader(
        image_size=(args.image_width, args.image_height),
        n_train_classses=args.n_train_classes,
        n_test_classes=args.n_test_classes
    )
    saver = tf.train.Saver()
    ckpt = tf.train.get_checkpoint_state(args.save_dir + '/' + args.model)
    with tf.Session() as sess:
        saver.restore(sess, ckpt.model_checkpoint_path)
        print("Test Result\n1st\t2nd\t3rd\t4th\t5th\t6th\t7th\t8th\t9th\t10th\tloss")
        y_list = []
        output_list = []
        loss_list = []
        for b in range(args.test_batch_num):
            x_image, x_label, y = data_loader.fetch_batch(args.n_classes, args.batch_size, args.seq_length,
                                                          type='test',
                                                          augment=args.augment,
                                                          label_type=args.label_type)
            feed_dict = {model.x_image: x_image, model.x_label: x_label, model.y: y}
            output, learning_loss = sess.run([model.o, model.learning_loss], feed_dict=feed_dict)
            y_list.append(y)
            output_list.append(output)
            loss_list.append(learning_loss)
        accuracy = test_f(args, np.concatenate(y_list, axis=0), np.concatenate(output_list, axis=0))
        for accu in accuracy:
            print('%.4f' % accu, end='\t')
        print(np.mean(loss_list))


def test_f(args, y, output):
    correct = [0] * args.seq_length
    total = [0] * args.seq_length
    if args.label_type == 'one_hot':
        y_decode = one_hot_decode(y)
        output_decode = one_hot_decode(output)
    elif args.label_type == 'five_hot':
        y_decode = five_hot_decode(y)
        output_decode = five_hot_decode(output)
    for i in range(np.shape(y)[0]):
        y_i = y_decode[i]
        output_i = output_decode[i]
        # print(y_i)
        # print(output_i)
        class_count = {}
        for j in range(args.seq_length):
            if y_i[j] not in class_count:
                class_count[y_i[j]] = 0
            class_count[y_i[j]] += 1
            total[class_count[y_i[j]]] += 1
            if y_i[j] == output_i[j]:
                correct[class_count[y_i[j]]] += 1
    #  return [float(correct[i]) / total[i] if total[i] > 0. else 0. for i in range(1, int(args.seq_length/args.n_classes))]
    return [float(correct[i]) / total[i] if total[i] > 0. else 0. for i in range(1, 8)], total[1:8]


if __name__ == '__main__':
    main()
