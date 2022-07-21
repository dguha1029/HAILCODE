import argparse
import os
import tensorflow as tf
from model import Model



"""
This script defines hyperparameters.
"""



def configure():
	flags = tf.compat.v1.app.flags

	# training
	flags.DEFINE_integer('num_steps', 2000, 'maximum number of iterations')
	flags.DEFINE_integer('save_interval', 1000, 'number of iterations for saving and visualization')
	flags.DEFINE_integer('random_seed', 1234, 'random seed')
	flags.DEFINE_float('weight_decay', 0.0005, 'weight decay rate')
	flags.DEFINE_float('learning_rate', 2.5e-4, 'learning rate')
	flags.DEFINE_float('power', 0.9, 'hyperparameter for poly learning rate')
	flags.DEFINE_float('momentum', 0.9, 'momentum')
	flags.DEFINE_string('encoder_name', 'deeplab', 'name of pre-trained model, res101, res50 or deeplab')
	flags.DEFINE_string('pretrain_file', '/Users/deyaa/OneDrive/Desktop/deeplab_resnet_tf/deeplab_resnet_init.ckpt', 'pre-trained model filename corresponding to encoder_name')
	flags.DEFINE_string('data_list', '/Users/deyaa/OneDrive/Desktop/Deeplab-v2--ResNet-101--Tensorflow-master/dataset/TrainSet.txt', 'training data list filename')

	# validation
	flags.DEFINE_integer('valid_step', 2000, 'checkpoint number for validation')
	flags.DEFINE_integer('valid_num_steps', 1449, '= number of validation samples')
	flags.DEFINE_string('valid_data_list', '/Users/deyaa/OneDrive/Desktop/Deeplab-v2--ResNet-101--Tensorflow-master/dataset/val.txt', 'validation data list filename')

	# prediction / saving outputs for testing or validation
	flags.DEFINE_string('out_dir', out_dir_, 'directory for saving outputs')
	flags.DEFINE_integer('test_step', test_step_, 'checkpoint number for testing/validation')
	flags.DEFINE_integer('test_num_steps', test_num_steps_, '= number of testing/validation samples')
	flags.DEFINE_string('test_data_list',test_data_list_, 'testing/validation data list filename')
	flags.DEFINE_boolean('visual', True, 'whether to save predictions for visualization')

	# data
	flags.DEFINE_string('data_dir', data_dir_, 'data directory')
	flags.DEFINE_integer('batch_size', 10, 'training batch size')
	flags.DEFINE_integer('input_height', 321, 'input image height')
	flags.DEFINE_integer('input_width', 321, 'input image width')
	flags.DEFINE_integer('num_classes', 21, 'number of classes')
	flags.DEFINE_integer('ignore_label', 255, 'label pixel value that should be ignored')
	flags.DEFINE_boolean('random_scale', True, 'whether to perform random scaling data-augmentation')
	flags.DEFINE_boolean('random_mirror', True, 'whether to perform random left-right flipping data-augmentation')
	
	# log
	flags.DEFINE_string('modeldir', modeldir_, 'model directory')
	flags.DEFINE_string('logfile', 'log.txt', 'training log filename')
	flags.DEFINE_string('logdir', 'log', 'training log directory')
	
	flags.FLAGS.__dict__['__parsed'] = False
	return flags.FLAGS

def main(_):
	if args.option not in ['train', 'test', 'predict']:
		print('invalid option: ', args.option)
		print("Please input a option: train, test, or predict")
	else:
		# Set up tf session and initialize variables.
		# config = tf.ConfigProto()
		# config.gpu_options.allow_growth = True
		# sess = tf.Session(config=config)
		sess = tf.Session()
		# Run
		model = Model(sess, configure(test_data_list_=args.test_data_list, out_dir_=args.out_dir, test_step_=args.test_step,
		test_num_steps_=args.test_num_steps, modeldir_=args.modeldir, data_dir_=args.data_dir,
		num_steps_=args.num_steps, save_interval_=args.save_interval, learning_rate_=args.learning_rate,
		pretrain_file_=args.pretrain_file, data_list_=args.data_list, batch_size_=args.batch_size,
		input_height_=args.input_height, input_width_=args.input_width, num_classes_=args.num_classes,
		print_color_=args.print_color, log_dir_=args.log_dir, log_file_=args.log_file, encoder_name_=args.encoder_name))
		getattr(model, args.option)()


if __name__ == '__main__':
	parser = argparse.ArgumentParser()

	parser.add_argument('--option', dest='option', type=str, default='train',
		help='actions: train, test, or predict')
	parser.add_argument('--test_data_list', dest='test_data_list', type=str, default='./dataset/test.txt',
		help='testing/validation data list filename')
	parser.add_argument('--out_dir', dest='out_dir', type=str, default='output',
		help='directory for saving testing outputs')
	parser.add_argument('--test_step', dest='test_step', type=int, default=350000,
		help='checkpoint number for testing/validation')
	parser.add_argument('--test_num_steps', dest='test_num_steps', type=int, default=1000,
		help='number of testing/validation samples')
	parser.add_argument('--modeldir', dest='modeldir', type=str, default='modelAugment',
		help='model directory')
	parser.add_argument('--data_dir', dest='data_dir', type=str, default='/hdd/wsi_fun/ImageAugCustom/AugmentationOutput',
		help='data directory')
	parser.add_argument('--gpu', dest='gpu', type=str, default='0',
		help='specify which GPU to use')
	parser.add_argument('--num_steps', dest='num_steps', type=int, default=100000,
		help='maximum number of iterations')
	parser.add_argument('--save_interval', dest='save_interval', type=int, default=15000,
		help='number of iterations for saving and visualization')
	parser.add_argument('--learning_rate', dest='learning_rate', type=float, default=2.5e-4,
		help='learning rate')
	parser.add_argument('--pretrain_file', dest='pretrain_file', type=str, default='deeplab_resnet.ckpt',
		help='pre-trained model filename corresponding to encoder_name')
	parser.add_argument('--data_list', dest='data_list', type=str, default='./dataAugment/train.txt',
		help='training data list filename')
	parser.add_argument('--batch_size', dest='batch_size', type=int, default=15,
		help='training batch size')
	parser.add_argument('--input_height', dest='input_height', type=int, default=256,
		help='input image height')
	parser.add_argument('--input_width', dest='input_width', type=int, default=256,
		help='input image width')
	parser.add_argument('--num_classes', dest='num_classes', type=int, default=2,
		help='number of classes in images')
	parser.add_argument('--log_dir', dest='log_dir', type=str, default="log",
		help='directory for saving log files')
	parser.add_argument('--log_file', dest='log_file', type=str, default="log.txt",
		help='Default logfile name')
	parser.add_argument('--print_color', dest='print_color', type=str, default="\033[0;37;40m",
		help='color of printed text')
	parser.add_argument('--encoder_name', dest='encoder_name', type=str, default=" ",
		help='color of printed text')


	args = parser.parse_args()

	# Choose which gpu or cpu to use
	os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
	tf.compat.v1.app.run()