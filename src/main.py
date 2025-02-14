from algorithm.train import Trainer
from model.translator import Translator
from data.data import TranslateData

import hydra
from omegaconf import DictConfig, OmegaConf

@hydra.main(version_base=None, config_path="config", config_name="config")
def main(config : DictConfig) -> None:
    ## DATA ##
    data = TranslateData(config)

    ## MODEL ##
    src_vocab, tgt_vocab = len(data.vocab['de']), len(data.vocab['en'])
    model = Translator(src_vocab, tgt_vocab, config)
    print('Model Created.')

    ## ALGORITHM ##
    print('Running Algorithm...')
    algorithm = Trainer(data, model, config)
    algorithm.run()
    print('Done!')

if __name__ == "__main__":
    main()
