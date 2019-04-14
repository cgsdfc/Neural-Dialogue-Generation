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
            print('months:', self.months, file=f, end='')
            return f.getvalue()


class Dataset:

    def __init__(self, name, lines):
        self.name = name
        self.lines = lines


class Statistics:

    def __init__(self, dataset, batch_size, batch_time):
        self.dataset = dataset
        self.batch_size = batch_size
        self.batch_time = batch_time

    def n_batch_loop_over(self):
        """
        How many batches we need to loop over the whole dataset?

        :return:
        """
        return self.dataset.lines / self.batch_size

    def epoch_time(self):
        """
        Time to train one epoch.
        Using Jiwei's model.

        :return:
        """
        nb = self.n_batch_loop_over()
        time = nb * self.batch_time
        return TimeUnit(time)


d2_3_train = Dataset('dialogue_length2_3_train', 33901902)

d2_3_test = Dataset('dialogue_length2_3_test', 4237738)

d2_6_train = Dataset('dialogue_length2_6_train', 21012179)
d2_6_test = Dataset('dialogue_length2_6_test', 2626522)

# Measured statistics.
atten_stats = {
    # dialogue_length 2 3.
    # 'd2_3_bs_20': Statistics(d2_3_train, 20, 0.6598),
    # 'd2_3_bs_40': Statistics(d2_3_train, 40, 0.9957),
    'd2_3_bs_60_train': Statistics(d2_3_train, 60, 1.1762),  # This is so far the best.
    # 'd2_3_bs_80': Statistics(d2_3_train, 80, 2.2099),  # This memory error.

    'd2_3_bs_60_test': Statistics(d2_3_test, 60, 1.1762),  # Nearly a day to run the test.

    # dialogue_length 2 6.
    'd2_6_bs_60_train': Statistics(d2_6_train, 60, 1.1624),  # 4.7 days.
    'd2_6_bs_65_train': Statistics(d2_6_train, 65, 1.3851),  # 5 days.

    'd2_6_bs_60_test': Statistics(d2_3_test, 60, 1.1624),  # less than one hour.

    'd2_6_bs_50_train': Statistics(d2_3_train, 50, 1.0584),  # 8.3 days.

}

if __name__ == '__main__':
    d2_3_bs_60_valid_freq = 5000
    batch_time = atten_stats['d2_3_bs_60_train'].batch_time
    valid_interval = d2_3_bs_60_valid_freq * batch_time
    valid_interval = TimeUnit(valid_interval)
    print('validate/save every', valid_interval.hours, 'hours')
    print()

    for name, stat in atten_stats.items():
        print(name)
        print(stat.epoch_time())
        print('n_batch_loop_over:', stat.n_batch_loop_over())
        print()

# d2_3_bs_20
# seconds: 1118423.74698
# hours: 310.67326305
# days: 12.944719293750001
# weeks: 1.8492456133928572
# months: 0.06164152044642857
#
# d2_3_bs_40
# seconds: 843903.0955350001
# hours: 234.41752653750004
# days: 9.7673969390625 weeks: 1.3953424198660715
# months: 0.04651141399553572
#
# d2_3_bs_60
# seconds: 664590.28554
# hours: 184.60841265
# days: 7.692017193749999
# weeks: 1.0988595991071428
# months: 0.03662865330357143
#
# d2_3_bs_80
# seconds: 936497.6653725001
# hours: 260.13824038125
# days: 10.839093349218752
# weeks: 1.54844190703125
# months: 0.051614730234375
#
# dialogue_length2_3_test
# seconds: 83073.79059333332
# hours: 23.07605294259259
# days: 0.9615022059413579
# weeks: 0.13735745799162255
# months: 0.004578581933054085
# n_batch_loop_over: 70628.96666666666

# dialogue_length2_6_train
# seconds: 407075.9478266667
# hours: 113.07665217407408
# days: 4.7115271739197535
# weeks: 0.6730753105599648
# months: 0.02243584368533216
# n_batch_loop_over: 350202.98333333334
#
# dialogue_length2_3_test
# seconds: 820.9911085333334
# hours: 0.22805308570370372
# days: 0.00950221190432099
# weeks: 0.001357458843474427
# months: 4.5248628115814234e-05
# n_batch_loop_over: 706.2896666666667
