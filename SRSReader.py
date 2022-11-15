import matplotlib.pyplot as plt 
import numpy as np 
import pandas as pd 
from datetime import datetime, timedelta
import struct
import matplotlib.ticker as ticker
from scipy.ndimage import gaussian_filter
from scipy.signal import find_peaks
import matplotlib.dates as mdates
from scipy.optimize import curve_fit
import sys
#plt.style.use('~/evanstyle.mplstyle')


class SRSReader:
    def __init__(self, infile, points_per_amu=20, min_mass=1):
        self.infile = infile
        self.data = [] #each index is a new scan, with its elements being dict with timestamp and new values
        self.ppa = points_per_amu
        self.min_mass = min_mass 
        #maximum mass is assumed by how many points are in the scan and points_per_amu



    #problem at the moment: it does not seem like
    #the min and max AMU nor the points per amu are recorded
    #in the binary (or I can't find them yet). SO i am compromizing
    #and assuming a min of 1 amu and 20 points per scan to infer the max AMU.
    #This is something to take note when setting up the continuous scanning. 
    def load_data(self):
        print("opening data file: " + self.infile)
        f = open(self.infile, 'rb')
        content = f.read()
        st = 0 #start byte
        db = 1 #read one byte at a time
        end = st + db #next byte
        marker_words = ['M', '\x00', '\x00', '\x80'] #picks out a start of scan, cleverly counting known number of relative bytes
        match_sequence_counter = 0 #counts how many bytes match the above sequence
        start_indices = [] #finds starting indices for scan data
        date_indices = [] #finds starting indices for datetime stamps
        print("Indexing scans")
        while True: #read until no more bytes left
            if(end >= len(content)):
                break
            data = struct.unpack("c",content[st:end]) #unpack single character
            st += db
            end += db
            data = [_.decode("ISO-8859-1") for _ in data] #decode in string form
            if(data[0] == marker_words[match_sequence_counter]): #if this byte is continuing to match the sequence
                match_sequence_counter += 1
            else:
                match_sequence_counter = 0 #reset counter
            if(match_sequence_counter == 4):
                #got a start scan marker
                start_indices.append(st + 22)
                date_indices.append(st - 27)
                match_sequence_counter = 0

        #make a list of scan dates
        date_length = 24 #bitlength for date chars
        scan_dates = [] #datetime objects
        date_format = "%b %d, %Y  %I:%M:%S %p"
        for didx in date_indices:
            date_str = content[didx-1:didx+date_length].decode("ISO-8859-1")
            scan_dates.append(datetime.strptime(date_str, date_format))




        #get scan data
        print("Grabbing pressures from each scan")
        min_amu = 1
        max_amu = None #to determine
        for i in range(len(start_indices)):
            scan_idx = start_indices[i]
            if(i + 1 < len(start_indices)):
                end_idx = date_indices[i+1] - 1 #the end of the scan (variable based on settings) is referenced to the date of next scan
            else:
                break #not sure yet how to handle last scan, this throws it out. 

            st = scan_idx
            db = 4
            end = st+db
            pps = []
            while True:
                if(end >= end_idx):
                    break

                data = struct.unpack('f', content[st:end])
                pps.append(data[0])
                st += db
                end += db

            self.data.append({"time":scan_dates[i], "pressures":pps})

        #here, the mass information is stored as two pieces of info,
        #and then for each scan is reconstructed at plot time or fit time
        #based on the number of data points in the pressures list. This allows
        #scans of different mass ranges to be in the same dataset, i.e. when chaining
        #datasets together.
        max_amu = int(len(pps)/self.ppa)
        for k in self.data:
            k["min_amu"] = self.min_mass
            k["max_amu"] = max_amu

        print("Done loading data")


    #operator overload addition of datasets. 
    def __add__(self, other):
        #infile gets turned into a list, breaking functionality of
        #the "load_data" or "debug_binary" system - this just adds the self.data
        #lists and makes sure scans are in time order. 
        s = SRSReader(infile=[self.infile, other.infile])
        my_t0 = self.data[0]["time"]
        other_t0 = other.data[0]["time"]
        if(my_t0 > other_t0):
            s.data = [*other.data, *self.data] #concatenation
        else:
            s.data = [*self.data, *other.data]

        return s


    def debug_binary(self, istart=None, iend=None):
        print("opening data file: " + self.infile)
        f = open(self.infile, 'rb')
        content = f.read()

        if(istart is None):
            istart = 0
            iend = 50 #default to 50 "words"

        #character sequence generator
        character_sequence = [] 
        st = 0 #start byte
        db = 1 #read one byte at a time
        end = st + db #next byte
        i = 0
        while True: #read until no more bytes left
            if(end >= len(content)):
                break
            if(i < istart):
                continue
            if(i > iend):
                break

            data = struct.unpack("c",content[st:end]) #unpack single character
            st += db
            end += db
            i += 1
            character_sequence.append(data[0].decode("ISO-8859-1"))

        #integer sequence generator
        integer_sequence = [] 
        st = 0 #start byte
        db = 4 #read one byte at a time
        end = st + db #next byte
        i = 0
        while True: #read until no more bytes left
            if(end >= len(content)):
                break
            if(i < istart):
                continue
            if(i > iend):
                break

            data = struct.unpack("i",content[st:end]) #unpack single character
            st += db
            end += db
            i += 1
            integer_sequence.append(data[0])

        #float sequence generator
        float_sequence = [] 
        st = 0 #start byte
        db = 4 #read one byte at a time
        end = st + db #next byte
        i = 0
        while True: #read until no more bytes left
            if(end >= len(content)):
                break
            if(i < istart):
                continue
            if(i > iend):
                break

            data = struct.unpack("f",content[st:end]) #unpack single character
            st += db
            end += db
            i += 1
            float_sequence.append(data[0])


        for i in range(len(character_sequence)):
            print(str(i)+ ":   ",end='')
            print(character_sequence[i], end = "\t")
            print(integer_sequence[i], end = "\t\t\t\t")
            print("{:.2e}".format(float_sequence[i]))



    def plot_scan(self, scan_idx, ax=None):
        if(ax == None):
            fig, ax = plt.subplots(figsize=(14, 8))
        e = self.data[scan_idx]
        masses = np.linspace(e["min_amu"], e["max_amu"], len(e["pressures"]))
        ax.set_title("Scan at: " + e["time"].strftime("%b %d, %Y  %H:%M:%S"))
        ax.plot(masses, e["pressures"])
        ax.set_xlabel("M/Q (amu)")
        ax.set_ylabel("Torr")
        ax.set_yscale('log')
        ax.set_ylim([1e-10, 2e-5])
        ax.set_xlim([1, max(masses)])
        ax.xaxis.set_major_locator(ticker.MultipleLocator(2))
        ax.xaxis.set_minor_locator(ticker.MultipleLocator(1))
        ax.grid(True)
        return fig, ax



    #mass is the desired mass to fit for the pressure-vs-time
    #window is a window of amu for the peak finding algorithm
    #fit is whether you want to fit a particular region (set by mintime and maxtime datetime objects)
    #duration is whether you want the primary x axis to be duration or datetimes
    def plot_mass_evolution_peakfound(self, mass, ax=None, window=5, fit=False, mintime=None, maxtime=None, duration=True):
        if(ax is None):
            fig, ax = plt.subplots(figsize=(14,8))


        ps = []
        ts = []
        ts_h = [] #time in hours since start
        peakfound_masses = []
        for i, event in enumerate(self.data):
            ts.append(event["time"])
            ts_h.append((ts[-1] - ts[0]).total_seconds()/3600)
            p, m = self.get_peakfound_pressure(mass, i, window=window)
            ps.append(p)
            peakfound_masses.append(m)

        if(duration):
            #make an x axis on top with date labels
            axd = ax.twiny()
            l = axd.plot(ts, ps)
            l.pop().remove() #remove the line from plot
            axd.xaxis.set_major_formatter(mdates.DateFormatter("%m/%d\n %H:%M"))
            axd.grid(False)


        mstd = np.std(peakfound_masses)
        massmean = np.mean(peakfound_masses)
        if(duration):
            ax.plot(ts_h, ps, '.',label=str(round(massmean, 2)) + " amu +- " + str(round(mstd, 2)))
            ax.set_xlabel("Hours since start")
        else:
            ax.plot(ts, ps, '.',label=str(round(massmean, 2)) + " amu +- " + str(round(mstd, 2)))
            ax.xaxis.set_major_formatter(mdates.DateFormatter("%m/%d\n %H:%M"))

        ax.set_ylabel("Torr")
        ax.set_yscale('log')
        


        def fitfunc(x, tau, A, b):
            return A*np.exp(-(x-b)/tau)/tau
        #if fitting, re-do but with a fit range
        if(fit):
            if(mintime is None or maxtime is None):
                mintime = ts[0]
                maxtime = ts[0]
            offset = (mintime-ts[0]).total_seconds()
            ps = []
            ts = []
            ts_s = [] #time in hours since start
            peakfound_masses = []
            for i, event in enumerate(self.data):
                if(mintime < event["time"] < maxtime):
                    ts.append(event["time"])
                    ts_s.append((ts[-1] - ts[0]).total_seconds()) #seconds now
                    p, m = self.get_peakfound_pressure(mass, i, window=window)
                    ps.append(p)
                    peakfound_masses.append(m)

            guess = [1e4, np.mean(ps), np.mean(ts_s)]
            popt, pcov = curve_fit(fitfunc, ts_s, ps, p0=guess)
            ax.plot([(_ + offset)/3600 for _ in ts_s], fitfunc(np.array(ts_s), *popt), 'k--')
            print("{:.1f} amu: {:.2e} e^(-(x - {:.2f})/{:.2f})/{:.2f}".format(np.mean(peakfound_masses), popt[1], popt[2]/3600, popt[0]/3600, popt[0]/3600))





        return ax

    #same as above but take ratio relative to a different mass
    def plot_mass_evolution_peakfound_relative(self, mass, mass_rel, ax=None, window=5):
        if(ax is None):
            fig, ax = plt.subplots(figsize=(14,8))


        ps = []
        ts = []
        ts_h = [] #time in hours since start
        peakfound_masses = []
        for i, event in enumerate(self.data):
            ts.append(event["time"])
            ts_h.append((ts[-1] - ts[0]).total_seconds()/3600)
            p, m = self.get_peakfound_pressure(mass, i, window=window)
            prel, mrel = self.get_peakfound_pressure(mass_rel, i, window=window)
            ps.append(p/prel)
            peakfound_masses.append(m)

        #make an x axis on top with date labels
        axd = ax.twiny()
        l = axd.plot(ts, ps)
        l.pop().remove() #remove the line from plot
        axd.xaxis.set_major_formatter(mdates.DateFormatter("%m/%d\n %H:%M"))
        axd.grid(False)


        mstd = np.std(peakfound_masses)
        massmean = np.mean(peakfound_masses)
        ax.plot(ts_h, ps, label=str(round(massmean, 2)) + " amu +- " + str(round(mstd, 2)))
        ax.set_ylabel("Torr")
        ax.set_yscale('log')
        ax.set_xlabel("Hours since start")
        #add an x axis on the top for dates

        return ax

    #find the peak of the pressure at mass
    #for scan i in the dataset by (1) gaussian
    #smoothing and (2) peakfinding then (3) determinig
    #which peak is closest to mass
    def get_peakfound_pressure(self, mass, i, window=5):
        #get pressures
        e = self.data[i]
        ms = np.linspace(e["min_amu"], e["max_amu"], len(e["pressures"]))
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


    #min/maxtimes are range to perform fit, datetime objects
    def fit_pumpdown(self, mass, mintime, maxtime):
        ps = []
        ts = []
        ts_h = [] #time in hours since start
        peakfound_masses = []
        for i, event in enumerate(self.data):
            if(mintime < event["time"] < maxtime):
                ts.append(event["time"])
                ts_s.append((ts[-1] - ts[0]).total_seconds()) #seconds now
                p, m = self.get_peakfound_pressure(mass, i, window=window)
                prel, mrel = self.get_peakfound_pressure(mass_rel, i, window=window)
                ps.append(p/prel)
                peakfound_masses.append(m)

        def fitfunc(x, tau, A, b):
            return A*np.exp(-(x-b)/tau)/tau

        guess = [1e4, np.mean(ps), np.mean(ts)]
        popt, pcov = curve_fit(fitfunc, ts_s, ps, p0=guess)








