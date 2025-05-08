import configparser

def change():
    try:
        config_tempesta = configparser.ConfigParser()
        config_tempesta.read('./config_tempesta/config_tempesta.conf')
        type_ai = config_tempesta['parameters-set-up'].get('aitype', None)

        type_ai_list = str(type_ai.lower().split()[0])

        config_method = configparser.ConfigParser()
        config_method.read("./method_analyze/method_"+str(type_ai_list)+"/config_method.txt")

        list_model = ','.join(config_method['config-method'].get('list_model', '').split(','))
        type_analsze_belly = config_method['config-method'].getint('type_analsze', None)
        method_analyze = config_method['config-method'].get('method_analyze', None)

        config_main = configparser.ConfigParser()
        config_main.read("config_set_up_api_analyze.txt")

        if 'parameters-set-up' in config_main:
            config_main['parameters-set-up']['list_model'] = list_model
            config_main['parameters-set-up']['type_analsze'] = str(type_analsze_belly)
            config_main['parameters-set-up']['method_analyze'] = str(method_analyze)

        with open('config_set_up_api_analyze.txt', 'w') as configfile:
            config_main.write(configfile, space_around_delimiters=False)
        print(f"Successfully checked ai type")
    except Exception as e:
        print(f"Successfully checked ai type error: {e}")
