import requests
import sys
import configparser

try:
    config_set_up = configparser.ConfigParser()
    config_set_up.read('config_set_up_api_analyze.txt')

    url_leader = config_set_up['send-status'].get('url_leader', None)
    port_leader = config_set_up['send-status'].getint('port_leader', None)
    send_leader = f"http://{url_leader}:{port_leader}/statusAi"

    status_leader = f"http://{url_leader}:{port_leader}/statusLeader"
    response_leader = requests.get(status_leader, timeout=5)

    data_health = configparser.ConfigParser()
    data_health.read('status_health.txt')
    status_health = data_health['main-health'].get('status_health', 'False').lower() == 'true'
    #if status_health == True:
    if status_health == True and response_leader.status_code == 200:
        exit(0)  # succeed
    else:
        exit(1)  # fail
except requests.exceptions.RequestException as e:
    exit(1)  # fail