import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 
import os 
import sys 
from tqdm.notebook import trange, tqdm
from datetime import datetime, timedelta
from scipy.ndimage import gaussian_filter
from scipy.signal import find_peaks
import re

class MKSReader:
	def __init__(self, infiles):
		self.infiles = infiles

		self.masses = []
		self.data = [] #each index is a new scan, with its elements being dict with timestamp and new values


	#many scans stored in a single file. Many lines
	#of a header. Then, there is a line that defines
	#the "columns" that represent the scan time, and the
	#masses of each measurement point. Every scan is on
	#a single line. 
	def load_data(self):
		for infile in self.infiles:
			if(os.path.isfile(infile) == False):
				print("Could not find file : " + infile)
				return

			#load in the file reader
			f = open(infile, 'r')
			flines = f.readlines()
			#header is always 0:57
			#column line is line 56
			columnline = flines[56] #has mass values

			#splitting up all the garbage
			columnline = columnline.split('"')
			masses = []
			for c in columnline:
				if(len(c.split(' ')) == 2):
					masses.append(float(c.split(' ')[-1]))
			masses = sorted(masses)
			#check if there are already masses, and if so, 
			#are they equal. 
			if(len(self.masses) == 0):
				self.masses = masses
			else:
				if(self.masses != masses):
					print("Cannot combine file " + infile + ", their masses are unequal.")
					continue #to next file. 

			flines = flines[57:] #the rest are individual scans

			datetime_format = "%m/%d/%Y %I:%M:%S %p"
			looper = tqdm(flines)
			counter = 0
			for event in looper:
				e = {}
				if(len(event) < 1000):
					#likely at the end of file, some spaces or new lines
					continue
				t = event.split('"')[1] #the date timestamp of the scan
				t = datetime.strptime(t, datetime_format)
				e["time"] = t 
				#get the pressure values
				p = re.findall(r"[^,\s]+", event)
				#ignore the timestamp and such
				p = p[4:-1] #last element is bad for some reason?
				if(len(p) != len(masses)):
					print("Something wierd happened with number of data points")
					print(len(p))
					print(len(masses))
					continue

				e["pressures"] = [float(_) for _ in p]
				self.data.append(e)



	#plot scan by index i
	def plot_scan_by_index(self, i, ylim=[1e-10, 1e-6], ax=None):
		if(ax is None):
			fig, ax = plt.subplots(figsize=(12,8))

		p = self.data[i]["pressures"]
		t = self.data[i]["time"]

		#integrate spectrum for total pressure
		ptot = np.trapz(p, self.masses)

		ax.plot(self.masses, p, label=str(t) + ", integral = " + "{0:0.2e}".format(ptot) + " Torr-amu")
		ax.set_xlabel("amu")
		ax.set_ylabel("Torr")
		ax.set_yscale('log')
		ax.set_ylim(ylim)
		ax.legend()
		return ax

	def get_number_of_scans(self):
		return len(self.data)


	#for an input mass, plot the evolution of the pressure.
	#do so in the most simple way possible, by finding the mass
	#measurement closest to the input value, and just plotting
	#that value's pressure over time. Another function does this
	#but with a peak finding algorithm or interpolation. 
	def plot_mass_evolution_simple(self, mass, ax=None):
		if(ax is None):
			fig, ax = plt.subplots(figsize=(12,8))


		#closest value in masses list
		idx, m = min(enumerate(self.masses), key=lambda x: abs(x[1] - mass))
		ps = []
		ts = []
		for event in self.data:
			ts.append(event["time"])
			ps.append(event["pressures"][idx])

		ax.plot(ts, ps, label=str(round(m, 2)) + " amu")
		ax.set_ylabel("Torr")
		ax.set_yscale('log')

		return ax


	def plot_mass_evolution_peakfound(self, mass, ax=None, window=5):
		if(ax is None):
			fig, ax = plt.subplots(figsize=(12,8))


		ps = []
		ts = []
		peakfound_masses = []
		for i, event in enumerate(self.data):
			ts.append(event["time"])
			p, m = self.get_peakfound_pressure(mass, i, window=window)
			ps.append(p)
			peakfound_masses.append(m)

		#plot standard deviation of the mass peak found
		mstd = np.std(peakfound_masses)
		massmean = np.mean(peakfound_masses)

		ax.plot(ts, ps, label=str(round(massmean, 2)) + " amu +- " + str(round(mstd, 2)))
		ax.set_ylabel("Torr")
		ax.set_yscale('log')

		return ax

	#find the peak of the pressure at mass
	#for scan i in the dataset by (1) gaussian
	#smoothing and (2) peakfinding then (3) determinig
	#which peak is closest to mass
	def get_peakfound_pressure(self, mass, i, window=5):
		#get pressures
		ms = self.masses 
		ps = self.data[i]["pressures"]

		#get data within a reasonable range of the mass. 
		mass_window = float(window) #amu window around the input value, 
		ps_w = []
		ms_w = []
		for j, m in enumerate(ms):
			if(mass - mass_window/2 <= m <= mass + mass_window/2):
				ms_w.append(m)
				ps_w.append(ps[j])
		#fig, ax = plt.subplots(figsize=(12, 8))
		#ax.plot(ms_w, ps_w)

		#gaussian smooth the data 
		mass_period = 0.1 #amu period of smoothing
		dm = abs(ms[0] - ms[1])
		smoothing_samps = mass_period/dm
		ps_sm = gaussian_filter(ps_w, smoothing_samps)
		#ax.plot(ms_w, ps_sm)

		#find all peaks. 
		#0.5 amu distance between peaks at least
		peaks, _ = find_peaks(ps_sm, distance=0.5/dm, height=5e-10)
		if(len(peaks) == 0):
			#no peaks found, just take value closest
			#closest value in masses list
			idx, m = min(enumerate(ms_w), key=lambda x: abs(x[1] - mass))
			return ps_sm[idx], m

		peak_masses = [ms_w[_] for _ in peaks]
		#ax.plot(peak_masses, ps_sm[peaks], 'o', markersize=15)

		#ax.set_yscale('log')
		#plt.show()


		#closest value in masses list
		idx, m = min(enumerate(peak_masses), key=lambda x: abs(x[1] - mass))
		peak_idx = peaks[idx]
		#return also the mass that it found the peak at. 
		return ps_sm[peak_idx], m 



		


	


