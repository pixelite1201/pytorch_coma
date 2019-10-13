import os
import configparser

def set_default_parameters(config):
    config.add_section('Input Output')
    config.set('Input Output', 'visualize', 'True')
    config.set('Input Output', 'data_dir', '')
    config.set('Input Output', 'checkpoint_dir', '')
    config.set('Input Output', 'visual_output_dir', '')
    config.set('Input Output', 'template_fname', '')

    config.add_section('Model Parameters')
    config.set('Model Parameters', 'eval', 'False')
    config.set('Model Parameters', 'checkpoint_file', '')
    config.set('Model Parameters', 'n_layers', '4')
    config.set('Model Parameters', 'z', '8')
    config.set('Model Parameters', 'downsampling_factors', '4, 4, 4, 4')
    config.set('Model Parameters', 'num_conv_filters', '16, 16, 16, 32, 32')
    config.set('Model Parameters', 'polygon_order', '6, 6, 6, 6, 6')
    config.set('Model Parameters', 'workers_thread', 6)
    config.set('Model Parameters', 'optimizer', 'sgd')

    config.add_section('Learning Parameters')
    config.set('Learning Parameters', 'batch_size', 16)
    config.set('Learning Parameters', 'learning_rate', 8e-3)
    config.set('Learning Parameters', 'learning_rate_decay', 0.99)
    config.set('Learning Parameters', 'weight_decay', 5e-4)
    config.set('Learning Parameters', 'epoch', 300)


def read_config(fname):
    if not os.path.exists(fname):
        print('Config not found %s' % fname)
        return

    config = configparser.RawConfigParser()
    config.read(fname)

    config_parms = {}
    config_parms['visualize'] = config.getboolean('Input Output', 'visualize')
    config_parms['data_dir'] = config.get('Input Output', 'data_dir')
    config_parms['checkpoint_dir'] = config.get('Input Output', 'checkpoint_dir')
    config_parms['template_fname'] = config.get('Input Output', 'template_fname')
    config_parms['visual_output_dir'] = config.get('Input Output', 'visual_output_dir')

    config_parms['eval'] = config.getboolean('Model Parameters', 'eval')
    config_parms['checkpoint_file'] = config.get('Model Parameters', 'checkpoint_file')
    config_parms['n_layers'] = config.getint('Model Parameters', 'n_layers')
    config_parms['z'] = config.getint('Model Parameters', 'z')
    config_parms['downsampling_factors'] =  [int(x) for x in config.get('Model Parameters', 'downsampling_factors').split(',')]
    config_parms['num_conv_filters'] = [int(x) for x in config.get('Model Parameters', 'num_conv_filters').split(',')]
    config_parms['polygon_order'] = [int(x) for x in config.get('Model Parameters', 'polygon_order').split(',')]
    config_parms['workers_thread'] = config.getint('Model Parameters', 'workers_thread')
    config_parms['optimizer'] = config.get('Model Parameters', 'optimizer')


    config_parms['batch_size'] = config.getint('Learning Parameters', 'batch_size')
    config_parms['learning_rate'] = config.getfloat('Learning Parameters', 'learning_rate')
    config_parms['learning_rate_decay'] = config.getfloat('Learning Parameters', 'learning_rate_decay')
    config_parms['weight_decay'] = config.getfloat('Learning Parameters', 'weight_decay')
    config_parms['epoch'] = config.getint('Learning Parameters', 'epoch')
    return config_parms

if __name__ == '__main__':
    pkg_path, _ = os.path.split(os.path.realpath(__file__))
    config_fname = os.path.join(pkg_path, 'default.cfg')

    print('Writing default config file - %s' % config_fname)
    with open(config_fname, 'w') as configfile:
        config = configparser.RawConfigParser()
        set_default_parameters(config)
        config.write(configfile)
        configfile.close()



