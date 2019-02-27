from configparser import ConfigParser
import json
import os
from ast import literal_eval

type_caster = {
    'int': int,
    'str': lambda x: x if x.lower() != 'none' else None,
    'float': float,
    'bool': lambda x: True if x == 'True' else False if x == 'False' else exec("raise Exception('invalid bool value: {}'.format(x))"),
    'dict': json.loads,
    'list': json.loads,
    'tuple': literal_eval
}

class Config:
    __non_const_vars = ['NAME', 'GPU_IDS', 'IMAGES_PER_GPU', 'BATCH_SIZE', 'MODE', 'BACKBONE_WEIGHTS']

    def __init__(self, name='Configuration'):
        self.NAME = name
        pass

    def __str__(self, depth=0, show_type=True):
        d = self.__dict__
        s = ''
        for key in sorted(d.keys()):
            if d[key].__class__.__name__ == 'Config':
                s += '[{}]\n'.format(key)
                s += d[key].__str__(depth + 1, show_type)
            elif key == 'NAME':
                continue
            else:
                if show_type:
                    typename = d[key].__class__.__name__
                    if typename == 'NoneType':
                        typename = 'str'
                    s += depth * '\t' + '{}-{} = {}\n'.format(typename, key, str(d[key]).replace('\"', '\''))
                else:
                    s += depth * '\t' + '{} = {}\n'.format(key, str(d[key]).replace('\"', '\''))
        return s

    class ConstError(TypeError): pass

    def __setattr__(self, key, value):
        if key in self.__dict__ and key not in Config.__non_const_vars:
            raise self.ConstError('Cannot change the value of the constant {} in Config object'.format(key))
        else:
            self.__dict__[key] = value

    def flatten(self):
        d = self.__dict__
        conf = Config()
        conf.NAME = self.NAME
        for key in d:
            if d[key].__class__.__name__ == 'Config':
                _d = d[key].__dict__
                for _key in _d:
                    vars(conf)[_key] = _d[_key]
            else:
                vars(conf)[key] = d[key]

        if hasattr(conf, 'IMAGES_PER_GPU') and hasattr(conf, 'GPU_IDS'):
            conf.BATCH_SIZE = conf.IMAGES_PER_GPU * len(conf.GPU_IDS)

        if hasattr(conf, 'NUM_CLASSES'):
            conf.IMAGE_META_SIZE = 1 + 3 + 3 + 4 + 1 + conf.NUM_CLASSES

        if hasattr(conf, 'IMAGE_WIDTH') and hasattr(conf, 'IMAGE_HEIGHT'):
            conf.IMAGE_SHAPE = (conf.IMAGE_HEIGHT, conf.IMAGE_WIDTH, 3)

        return conf

    def display(self):
        print('\n-----------------{}-----------------\n'.format(self.NAME))
        print(self.__str__(show_type=False))
        print('\n-----------------{}-----------------\n'.format(len(self.NAME) * '-'))

    def save(self, path):
        f = open(path, 'w')
        f.write(self.__str__())
        f.close()

def load_config(filename):
    parser = ConfigParser()
    assert parser.read(filename), 'Could not read the file {}'.format(filename)

    config = Config()
    filename = os.path.basename(filename)
    config.NAME = filename[:-4] if filename.endswith('.cfg') else filename

    for section_name in parser.sections():
        vars(config)[section_name] = Config()
        section = parser.items(section_name)
        for item in section:
            _type_name = item[0]
            _type_name = _type_name.split('-')
            assert len(_type_name) == 2, 'invalid variable syntex: {}. Variables in .cfg file should be declared in the form type-VARIABLE_NAME'.format(item[0].upper())
            _type = _type_name[0].strip(' ')
            _name = _type_name[1].strip(' ').upper()
            comment_p = item[1].find('#')
            _value = item[1].strip(' ') if comment_p == -1 else item[1][:comment_p].strip(' ')
            vars(getattr(config, section_name))[_name] = type_caster[_type](_value)

    return config
