import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
from test_summary import test_dict

test_type_dict = {
    "Z-Test" : {
        "Distribution" : "Normal",
        "Score" : "Z-Score",
        'x' : lambda : np.linspace(-stats.norm.ppf(0.999), stats.norm.ppf(0.999), 1000),
        "pdf" : lambda x: stats.norm.pdf(x),
        "cdf" : lambda x: stats.norm.cdf(x),
        "sf" : lambda x: stats.norm.sf(x),
    },
    "T-Test" : {
        "Distribution" : r"Student's t",
        "Score" : "T-Score",
        'x' : lambda dof: np.linspace(-stats.t.ppf(0.999, dof), stats.t.ppf(0.999, dof), 1000),
        "pdf" : lambda x, dof: stats.t.pdf(x, dof),
        "cdf" : lambda x, dof: stats.t.cdf(x, dof),
        "sf" : lambda x, dof: stats.t.sf(x, dof),
    },
}

class Test_Statistic:
    def __init__(self):
        self.test_statistic_clear()

    def which_test(self, test_dict, test_name=None, test_type="Z-Test"):
        self.samples_info = self.samples_list[0] if self.num_sample == 1 else self.samples_list
        self.populations_info = self.populations_list[0] if self.num_population == 1 else self.populations_list
        if test_name is None:
            if test_type == "Z-Test":
                if self.num_sample == 1 and self.num_population == 1:
                    if self.populations_info.dist_type == "Normal":
                        test_key = "Test_1"
        else:
            test_key = test_name
        self.test = test_dict[test_key]
        print(f"Carry out test {test_key}...")

    def test_statistic(self):
        assert self.test and self.samples_info and self.populations_info
        return self.test(self.samples_info, self.populations_info)

    def time_value_push(self, times_list):
        self.times_list = times_list
        assert len(self.times_list) == len(self.samples_list)

    def calculate_dof(self):
        assert len(self.samples_list) == 2
        sample1, sample2 = self.samples_list
        n1, n2 = samples1.size, sample2.size
        diff1 = sample1.sample_data - sample1.mean
        diff2 = sample2.sample_data - sample2.mean
        s1_2, s2_2 = diff1 @ diff1 / (n1 - 1.), diff2 @ diff2 / (n2 - 1.)
        return np.round(
            (s1_2 / n1 + s2_2 / n2)**2. / (s1_2*s1_2 / (n1*n1*(n1-1.)) + s2_2*s2_2 / (n2*n2*(n2-1.)))
        )

    def sample_push(self, samples):
        for sample in samples:
            sample = sample if isinstance(sample, Sample) else Sample(sample)
            self.samples_list.append(sample)
        self.num_sample = len(self.samples_list)

    def population_push(self, populations):
        for population in populations:
            assert isinstance(population, Population)
            self.populations_list.append(population)
        self.num_population = len(self.populations_list)

    def test_statistic_clear(self):
        self.sample_clear()
        self.population_clear()
    
    def sample_clear(self):
        self.samples_list = []
        self.num_sample = 0

    def population_clear(self):
        self.populations_list = []
        self.num_population = 0

class Hypothesis_Test:
    def __init__(self, criterion=0.05, is_two_tailed=True, test_type="Z-Test", dof=0):
        self.criterion = criterion
        self.is_two_tailed = is_two_tailed
        self.test_type = test_type
        self.distribution_name = test_type_dict[self.test_type]["Distribution"]
        self.score_name = test_type_dict[self.test_type]["Score"]
        self.pdf_func = test_type_dict[self.test_type]["pdf"]
        self.cdf_func = test_type_dict[self.test_type]["cdf"]
        self.sf_func = test_type_dict[self.test_type]["sf"]
        if self.test_type == "Z-Test":
            self.x = test_type_dict[self.test_type]['x']()
            self.pdf = self.pdf_func(self.x)
            self.cdf = self.cdf_func(self.x)
        elif self.test_type == "T-Test":
            self.dof = dof
            assert self.dof > 0
            self.x = test_type_dict[self.test_type]['x'](self.dof)
            self.pdf = self.pdf_func(self.x, self.dof)
            self.cdf = self.cdf_func(self.x, self.dof)

    def xx_test(self, test_statistic, test_dict=test_dict, test_name=None):
        assert isinstance(test_statistic, Test_Statistic)
        if test_name is None:
            test_statistic.which_test(self.test_type)
        else:
            test_statistic.which_test(test_dict, test_name, self.test_type)
        self.score = test_statistic.test_statistic()
        if self.test_type == "Z-Test":
            self.test = self.pdf_func(self.score)
            self.pval = self.sf_func(abs(self.score)) * 2 if self.is_two_tailed else self.sf_func(abs(self.score))

        print(f"p-value = {self.pval:.4f}")
        if self.pval > self.criterion:
            print("Samples are likely drawn from the original distribution. FAIL TO REJECT H0")
        else:
            print("Samples are likely drawn from a different distribution. REJECT H0")
        if self.pval < 0.001:
            print("Statistical siginificance: p*** (p < 0.001)")
        elif self.pval < 0.01:
            print("Statistical siginificance: p** (p < 0.01)")
        elif self.pval < 0.05:
            print("Statistical siginificance: p* (p < 0.05)")
        else:
            print("Statistically insiginificant...")

    def plot_hypothesis_test(self, ax=None):
        if ax is None:
            fig, ax = plt.subplots(1, 1, figsize=(6, 6))
        ax.plot(self.x, self.pdf, 'g', label=f"{self.distribution_name} Distribution")
        ax.plot(self.x, self.cdf, 'r', alpha=0.5, label="Cumulative Distribution")
        if self.is_two_tailed:
            abs_score = abs(self.score)
            colored_region1 = self.x > abs_score
            colored_region2 = self.x < -abs_score
            ax.fill_between(self.x, self.pdf, where=colored_region1, facecolor='b', alpha=0.6, interpolate=True)
            ax.fill_between(self.x, self.pdf, where=colored_region2, facecolor='b', alpha=0.6, interpolate=True)
            ax.axvline(x=abs_score, alpha=0.8, ls='-.')
            ax.axvline(x=-abs_score, alpha=0.8, ls='-.')
        else:
            colored_region = self.x > self.score if self.score > 0 else self.x < self.score
            ax.fill_between(self.x, 0, self.pdf, where=colored_region, facecolor='b', alpha=0.4, interpolate=True)
            ax.axvline(x=self.score, alpha=0.8, ls='-.')
        ax.annotate(
            f"{self.score_name} = {self.score:.2f}, p-value = {self.pval:.4f}",
            xy=(self.score, self.test),
            xytext=(self.score * 0.5, self.test * 1.2)
        )
        ax.legend()
        ax.set_title(f"{self.distribution_name} Probability Density Function")
        ax.set_xlabel('x')
        ax.set_ylabel("p(x)")
        ax.set_ylim(0.0, 1.0)

class Sample:
    def __init__(self, sample_data):
        self.sample_data = sample_data
        self.mean = self.sample_data.mean()
        self.std = self.sample_data.std()
        self.var = self.std * self.std
        self.size = self.sample_data.size

class Population:
    def __init__(self, population_mean=None, population_std=None, dist_type="Normal"):
        self.mean = population_mean
        self.std = population_std
        self.var = None if self.std is None else self.std * self.std 
        self.dist_type = dist_type