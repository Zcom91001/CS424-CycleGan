import hashlib
import random
from pathlib import Path


def _file_to_scalar(path):
    digest = hashlib.md5(str(path).encode("utf-8")).hexdigest()
    return (int(digest[:8], 16) % 10000) / 10000.0


class UnpairedScalarDataset:
    """Lightweight domain A/B dataset that maps file paths to scalar features."""

    def __init__(self, domain_a_dir, domain_b_dir):
        self.a_files = sorted([p for p in Path(domain_a_dir).glob("**/*") if p.is_file()])
        self.b_files = sorted([p for p in Path(domain_b_dir).glob("**/*") if p.is_file()])
        if not self.a_files or not self.b_files:
            raise ValueError("Both domain directories must contain files.")

        self.a_values = [_file_to_scalar(p) for p in self.a_files]
        self.b_values = [_file_to_scalar(p) for p in self.b_files]

    def sample_batch(self, batch_size):
        a = random.choices(self.a_values, k=batch_size)
        b = random.choices(self.b_values, k=batch_size)
        return a, b

    def sample_preview(self, count=8):
        return (
            random.choices(self.a_values, k=count),
            random.choices(self.b_values, k=count),
        )

    def size_summary(self):
        return {
            "domain_a_files": len(self.a_files),
            "domain_b_files": len(self.b_files),
        }
