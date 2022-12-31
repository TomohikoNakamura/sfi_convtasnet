from typing import TextIO
import numpy

class Logger:
    """
    Logs the train progress to stdout and log file
    """
    print=print

    def multi_stage_to_text(self, losses):
        losses = [f"{l:4.2f}" for l in losses]
        return '->'.join(losses)

    def log_train_progress(self, epoch: int, losses, num_stages: int, learning_rate: float, progress: int) -> None:
        self.print('\r' + 250 * ' ', end='')  # clear line
        d, r = progress // 5, progress % 5
        loading_bar = d*'#' + (('.' if r < 1 else ':' if r < 2 else '|' if r < 3 else ')' if r < 4 else '>') + max(0, 19 - d)*'.' if progress < 100 else '')

        drums = self.multi_stage_to_text([losses[0 + i*4] for i in range(num_stages)])
        bass = self.multi_stage_to_text([losses[1 + i*4] for i in range(num_stages)])
        other = self.multi_stage_to_text([losses[2 + i*4] for i in range(num_stages)])
        vocals = self.multi_stage_to_text([losses[3 + i*4] for i in range(num_stages)])
        reconstruction = losses[0 + num_stages*4]
        dissimilarity = losses[1 + num_stages*4]
        similarity = losses[2 + num_stages*4]

        self.print(f'\repoch {epoch:3d} || train SDR drum {drums} | bass {bass} | other {other} | vocal {vocals} | mix {reconstruction:3.2f} | dis {dissimilarity:.3f} | sim {similarity:.3f} | lr {learning_rate:.3e} || {loading_bar} {progress:2d} %', end='', flush=True)

    def log_train(self, epoch: int, losses, num_stages: int, time_seconds: int, out_file: TextIO) -> None:
        self.print('\r' + 250 * ' ', end='')  # clear line

        drums = self.multi_stage_to_text([losses[0 + i*4] for i in range(num_stages)])
        bass = self.multi_stage_to_text([losses[1 + i*4] for i in range(num_stages)])
        other = self.multi_stage_to_text([losses[2 + i*4] for i in range(num_stages)])
        vocals = self.multi_stage_to_text([losses[3 + i*4] for i in range(num_stages)])
        reconstruction = losses[0 + num_stages * 4]
        dissimilarity = losses[1 + num_stages * 4]
        similarity = losses[2 + num_stages * 4]

        output = f"epoch {epoch:3d} || train SDR drum {drums} | bass {bass} | other {other} | vocal {vocals} | mix {reconstruction:3.2f} | dis {dissimilarity:.3f} | sim {similarity:.3f} || elapsed {time_seconds//60:02d}:{time_seconds%60:02d} || "
        self.print(f"\r{output}", end='', flush=True)
        self.print(f"{output}", end='', flush=True, file=out_file)

    def log_dev(self, losses, num_stages: int, learning_rate: float, out_file: TextIO) -> None:
        drums = self.multi_stage_to_text([losses[0 + i*4] for i in range(num_stages)])
        bass = self.multi_stage_to_text([losses[1 + i*4] for i in range(num_stages)])
        other = self.multi_stage_to_text([losses[2 + i*4] for i in range(num_stages)])
        vocals = self.multi_stage_to_text([losses[3 + i*4] for i in range(num_stages)])
        average = numpy.array(losses[-4:]).mean()

        output = f"dev SDR drum {drums} | bass {bass} | other {other} | vocal {vocals} | average {average:4.2f} || lr {learning_rate:.3e}"
        self.print(f"{output}", flush=True)
        self.print(f"{output}", flush=True, file=out_file)

    def log_dev_with_es(self, losses, best_loss, num_stages: int, learning_rate: float, num_bad_epochs: int, out_file: TextIO, elapsed_time: int=None) -> None:
        drums = self.multi_stage_to_text([losses[0 + i*4] for i in range(num_stages)])
        bass = self.multi_stage_to_text([losses[1 + i*4] for i in range(num_stages)])
        other = self.multi_stage_to_text([losses[2 + i*4] for i in range(num_stages)])
        vocals = self.multi_stage_to_text([losses[3 + i*4] for i in range(num_stages)])
        average = numpy.array(losses[-4:]).mean()

        if elapsed_time is None:
            output = f"dev SDR drum {drums} | bass {bass} | other {other} | vocal {vocals} | average {average:4.2f} || lr {learning_rate:.3e} | best {best_loss:.3e} | #bad epochs {num_bad_epochs:2d}"
        else:
            output = f"dev SDR drum {drums} | bass {bass} | other {other} | vocal {vocals} | average {average:4.2f} || lr {learning_rate:.3e} | best {best_loss:.3e} | #bad epochs {num_bad_epochs:2d} || elapsed {elapsed_time//60:02d}:{elapsed_time%60:02d}"
        self.print(f"{output}", flush=True)
        self.print(f"{output}", flush=True, file=out_file)

