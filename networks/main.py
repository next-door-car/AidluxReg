from base.base_config import BaseConfig
from .ad_net import ADNet
from .pe_net import PENet

def build_network(cfg: BaseConfig, net_name='ADNet') -> ADNet:
    """Builds the neural network."""

    implemented_networks = ('ADNet')
    assert net_name in implemented_networks
    net : ADNet
    if net_name == 'ADNet':
        net = ADNet(cfg)
    return net

def build_pretrain_encoder(cfg: BaseConfig, net_name='PENet') -> PENet:
    """Builds the neural network."""

    implemented_networks = ('PENet')
    assert net_name in implemented_networks
    net : PENet
    if net_name == 'PENet':
        net = PENet(cfg)
    return net