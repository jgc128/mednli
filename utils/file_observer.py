from pathlib import Path

import logging
from sacred.commandline_options import CommandLineOption
from sacred.observers import RunObserver

from utils.io import save_pickle


class FileObserver(RunObserver):
    def __init__(self, directory):
        self.base_dir = Path(directory)

        self.id = None
        self.config = None
        self.result = None
        self.repository = None
        self.name = None

        logging.info('Observing: %s', self.base_dir)

    def __get_experiment_id(self):
        exp_ids = [
            int(exp_file.stem.split('_')[1])
            for exp_file in self.base_dir.iterdir()
            if exp_file.is_file() and exp_file.name.startswith('exp_')
        ]

        if len(exp_ids) > 0:
            max_exp_id = max(exp_ids)
        else:
            max_exp_id = 0

        current_exp_id = max_exp_id + 1

        return current_exp_id

    def _get_experiment_filename(self, exp_id):
        return self.base_dir / 'exp_{}.pkl'.format(exp_id)

    def started_event(self, ex_info, command, host_info, start_time, config, meta_info, _id):
        current_exp_id = self.__get_experiment_id()
        self.id = current_exp_id

        # create an empty file for this experiment so the next experiments will see this one
        filename = self._get_experiment_filename(current_exp_id)
        open(str(filename), 'a').close()

        self.name = ex_info['name']
        self.repository = ex_info['repositories'][0]
        self.config = config

        return current_exp_id

    def completed_event(self, stop_time, result):
        self.result = result

        exp_results = {
            'id': self.id,
            'name': self.name,
            'repository': self.repository,
            'config': self.config,
            'result': self.result,
        }

        filename = self._get_experiment_filename(self.id)
        save_pickle(filename, exp_results)


class FileObserverDbOption(CommandLineOption):
    """Add a MongoDB Observer to the experiment."""

    short_flag = 'Z'
    arg = 'DIR'
    arg_description = 'Directory'

    @classmethod
    def apply(cls, args, run):
        file_observer = FileObserver(args)
        run.observers.append(file_observer)
