import io


class TimeUnit:
    SECOND = 1
    MINUTE = 60 * SECOND
    HOUR = MINUTE * 60
    DAY = HOUR * 24
    WEEK = DAY * 7
    MONTH = WEEK * 30

    def __init__(self, second):
        self.seconds = second

    @property
    def hours(self):
        return self.seconds / self.HOUR

    @property
    def days(self):
        return self.seconds / self.DAY

    @property
    def weeks(self):
        return self.seconds / self.WEEK

    @property
    def months(self):
        return self.seconds / self.MONTH

    def __str__(self):
        with io.StringIO() as f:
            print('seconds:', self.seconds, file=f)
            print('hours:', self.hours, file=f)
            print('days:', self.days, file=f)
            print('weeks:', self.weeks, file=f)
            print('months:', self.months, file=f)
            return f.getvalue()


class Dataset:

    def __init__(self, name, lines):
        self.name = name
        self.lines = lines


d2_3 = Dataset('dialogue_length2_3', 33901902)


class Statistics:

    def __init__(self, dataset, batch_size, batch_time):
        self.dataset = dataset
        self.batch_size = batch_size
        self.batch_time = batch_time

    def n_batch_loop_over(self):
        return self.dataset.lines / self.batch_size

    def epoch_time(self):
        nb = self.n_batch_loop_over()
        time = nb * self.batch_time
        return TimeUnit(time)


atten_stats = {
    'd2_3_bs_20': Statistics(d2_3, 20, 0.6598)
}

if __name__ == '__main__':
    for key, value in atten_stats.items():
        print(key)
        print(value.epoch_time())
