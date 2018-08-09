import random
import torch.utils.data.sampler


class BalancedBatchSampler(torch.utils.data.sampler.BatchSampler):
    def __init__(
        self,
        dataset_labels,
        batch_size=1,
        steps=None,
        n_classes=0,
        n_samples=2
    ):
        """ Create a balanced batch sampler for label based datasets
        Args
            dataset_labels : Labels of every entry from a dataset (in the same sequence)
            batch_size     : batch_size no explaination needed
            step_size      : Number of batches to generate (if None, then dataset_size / batch_size will be used)
            n_classes      : Number of classes
            n_samples      : Number of samples per class

            *** If batch_size > n_classes * n_samples, rest of batch will be randomly filled
        """
        self.batch_size = batch_size
        self.steps = len(dataset_labels) // batch_size if steps is None else steps
        self.n_classes  = n_classes
        self.n_samples  = n_samples

        # Create a label_to_entry_ids table
        self.label_to_entry_ids = {}
        for entry_id, label in enumerate(dataset_labels):
            if label in self.label_to_entry_ids:
                self.label_to_entry_ids[label].append(entry_id)
            else:
                self.label_to_entry_ids[label] = [entry_id]

        # Subset the labels with more than n_samples entries
        self.labels_subset = [label for (label, entry_ids) in self.label_to_entry_ids.items() if len(entry_ids) >= n_samples]
        assert len(self.labels_subset) >= n_classes, 'Too little labels have {} entries, choose a smaller n_classes or n_samples'.format(n_samples)

    def _make_batch_ids(self):
        batch_ids = []

        # Choose classes and entries
        labels_choosen = random.sample(self.labels_subset, self.n_classes)

        # Randomly sample n_samples entries from choosen labels
        for l in labels_choosen:
            batch_ids += random.sample(self.label_to_entry_ids[l], self.n_samples)

        if len(batch_ids) < self.batch_size:
            # Randomly sample remainder
            labels_choosen = {l: None for l in labels_choosen}
            remaining_entry_ids = []
            for label, entry_ids in self.label_to_entry_ids.items():
                if label not in labels_choosen:
                    remaining_entry_ids += entry_ids
            batch_ids += random.sample(remaining_entry_ids, self.batch_size - len(batch_ids))

        # Randomly shuffle batch ids
        batch_ids = random.sample(batch_ids, self.batch_size)
        batch_ids = torch.ByteTensor(batch_ids)

        return batch_ids

    def __iter__(self):
        self.count = 0
        while self.count < self.steps:
            self.count += 1
            yield self._make_batch_ids()

    def __len__(self):
        return self.steps
