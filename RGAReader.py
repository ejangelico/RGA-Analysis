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
import os
import re
import seaborn as sns
from scipy.interpolate import interp1d
#plt.style.use('~/evanstyle.mplstyle')


class RGAReader:
    def __init__(self, infile, rga_kind, points_per_amu=20, min_mass=1, pump_start = None):
        self.infile = infile
        self.data = [] #each index is a new scan, with its elements being dict with timestamp and new values
        self.ppa = points_per_amu
        self.min_mass = min_mass 
        #maximum mass is assumed by how many points are in the scan and points_per_amu
        self.rga_kind = rga_kind
        self.pump_start = pump_start #datetime of when the pump was started (not necessarily first scan start time)
        


    def load_data(self):
        if('mks' in self.rga_kind.lower()):
            self.load_data_mks()
        else:
            self.load_data_srs()

        if(self.pump_start is None):
            self.pump_start = self.data[0]["time"]
        else:
            self.pump_start = datetime.strptime(self.pump_start, "%Y-%m-%d %H:%M") 


    #many scans stored in a single file. Many lines
	#of a header. Then, there is a line that defines
	#the "columns" that represent the scan time, and the
	#masses of each measurement point. Every scan is on
	#a single line. 
    def load_data_mks(self):
        if(os.path.isfile(self.infile) == False):
            print("Could not find file : " + infile)
            return

        #load in the file reader
        f = open(self.infile, 'r')
        flines = f.readlines()
        #the line with all of the masses is always 1 line after the only line
        #that contains "Scan Data". 
        scandata_line = None
        for i in range(100):
            if("Scan Data" in flines[i]):
                scandata_line = i
                break

        columnline = flines[scandata_line + 1] #has mass values

        #splitting up all the garbage
        columnline = columnline.split('"')
        masses = []
        for c in columnline:
            if(len(c.split(' ')) == 2):
                masses.append(float(c.split(' ')[-1]))
        masses = sorted(masses)
        
        #compress the mass information into points per amu and min mass, to be
        #consistent with the SRS reader. 
        self.ppa = np.abs(masses[0] - masses[1])
        self.min_mass = np.min(masses)

        flines = flines[scandata_line + 2:] #the rest are individual scans

        datetime_format = "%m/%d/%Y %I:%M:%S %p"
        counter = 0
        for event in flines:
            print("On scan {:d} of {:d}".format(counter, len(flines)), end='\r')
            if(len(event) < 1000):
                #likely at the end of file, some spaces or new lines
                continue
            t = event.split('"')[1] #the date timestamp of the scan
            t = datetime.strptime(t, datetime_format)

            #get the pressure values
            p = re.findall(r"[^,\s]+", event)
            #ignore the timestamp and such
            p = p[4:-1] #last element is bad for some reason?
            if(len(p) != len(masses)):
                print("Something wierd happened with number of data points")
                print(len(p))
                print(len(masses))
                print(self.infile)
                continue

            self.data.append({"time":t, "pressures": [float(_) for _ in p], "min_amu":np.min(masses), "max_amu": np.max(masses)})
            counter += 1



    #problem at the moment: it does not seem like
    #the min and max AMU nor the points per amu are recorded
    #in the binary (or I can't find them yet). SO i am compromizing
    #and assuming a min of 1 amu and 20 points per scan to infer the max AMU.
    #This is something to take note when setting up the continuous scanning. 
    def load_data_srs(self):
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
            if(didx == np.min(date_indices)):
                print("First scan in this file is " + date_str)
            scan_dates.append(datetime.strptime(date_str, date_format))




        #get scan data
        print("Grabbing pressures from each scan")
        min_amu = 1
        max_amu = None #to determine
        pps = []
        for i in range(len(start_indices)):
            scan_idx = start_indices[i]
            if(i + 1 < len(start_indices)):
                end_idx = date_indices[i+1] - 1 #the end of the scan (variable based on settings) is referenced to the date of next scan
            else:
                #break #not sure yet how to handle last scan, this throws it out. 
                #or attempt to handle it by setting end_idx to be last packet in file
                end_idx = len(content) #this does seem to work. 


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
        if(len(pps) == 0):
            print("Error! No data found in file " + self.infile)
            return
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
        s = RGAReader(infile=[self.infile, other.infile], rga_kind=self.rga_kind)
        my_t0 = self.data[0]["time"]
        other_t0 = other.data[0]["time"]
        if(my_t0 > other_t0):
            s.data = [*other.data, *self.data] #concatenation
            s.pump_start = other.pump_start
        else:
            s.data = [*self.data, *other.data]
            s.pump_start = self.pump_start
        

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



    def plot_scan(self, scan_idx, ax=None, label=None):
        if(ax == None):
            fig, ax = plt.subplots(figsize=(14, 8))
        e = self.data[scan_idx]
        masses = np.linspace(e["min_amu"], e["max_amu"], len(e["pressures"]))
        ax.set_title("Scan at: " + e["time"].strftime("%b %d, %Y  %H:%M:%S"))
        if(label == None):
            ax.plot(masses, e["pressures"], linewidth=0.5)
        else:
            ax.plot(masses, e["pressures"], label=label, linewidth=0.5)
        ax.set_xlabel("M/Q (amu)")
        ax.set_ylabel("Torr")
        ax.set_yscale('log')
        ax.set_ylim([1e-10, 2e-5])
        ax.set_xlim([1, max(masses)])
        ax.xaxis.set_major_locator(ticker.MultipleLocator(2))
        ax.xaxis.set_minor_locator(ticker.MultipleLocator(1))
        ax.grid(True)
        return ax
    
    #picks out the value of partial pressure at the closest mass value 
    #as a function of time throughout the dataset. Does not fit the peaks,
    #though a function below will do that. 
    def plot_mass_evolution(self, mass, ax=None, mintime=None, maxtime=None, duration=True, dates=True, label=None, fit=False, smoothing=False):
        if(ax is None):
            fig, ax = plt.subplots(figsize=(14,8))


        ps = []
        ts = []
        ts_h = [] #time in hours since start
        #in case the mass point does not lie within the data,
        #give a window of which to say "not in data"
        mass_exclusion = 0.5 #amu
        for i, event in enumerate(self.data):
            data_masses = np.linspace(event["min_amu"], event["max_amu"], len(event["pressures"]))
            m_idx = (np.abs(np.array(data_masses) - mass)).argmin()
            m_returned = data_masses[m_idx]
            if(np.abs(m_returned - mass) > mass_exclusion):
                p = None
                continue #no data at this mass region
            
            p = event["pressures"][m_idx]
            ps.append(p)
            ts.append(event["time"])
            ts_h.append((ts[-1] - self.pump_start).total_seconds()/3600)

        if(duration and dates):
            #make an x axis on top with date labels
            axd = ax.twiny()
            l = axd.plot(ts, ps)
            l.pop().remove() #remove the line from plot
            axd.xaxis.set_major_formatter(mdates.DateFormatter("%m/%d\n %H:%M"))
            axd.grid(False)

        if(label is None):
            label = str(round(m_returned, 2))

        filt_n = 10 #samples filter kernel 

        if(duration):
            l = ax.plot(ts_h, ps, '.',label=label)
            color = l[0].get_color()
            if(smoothing):
                ax.plot(ts_h, gaussian_filter(ps, filt_n), '-', color=color)
            ax.set_xlabel("Hours since start")
        else:
            l = ax.plot(ts, ps, '.',label=label)
            color = l[0].get_color()
            if(smoothing):
                ax.plot(ts, gaussian_filter(ps, filt_n), '-', color=color)
            ax.xaxis.set_major_formatter(mdates.DateFormatter("%m/%d\n %H:%M"))


        def fitfunc(x, al, A):
            return A*x**(-al)
        
        #if fitting, re-do but with a fit range
        if(fit):
            if(mintime is None or maxtime is None):
                mintime = ts[0]
                maxtime = ts[-1]
            offset = (mintime-ts[0]).total_seconds()
            ps_fit = []
            ts_fit = []
            ts_fit_s = [] #time in seconds
            for fidx, t in enumerate(ts):
                if(mintime <= t and maxtime >= t):
                    ps_fit.append(ps[fidx])
                    ts_fit.append(t)
                    ts_fit_s.append(ts_h[fidx]*3600)


            guess = [1, 1e-2]
            popt, pcov = curve_fit(fitfunc, ts_fit_s, ps_fit, p0=guess)
            ax.plot(np.array(ts_fit_s)/3600, fitfunc(np.array(ts_fit_s), *popt), '--', color=color, label="({:.2e})t^(-{:.2f})".format(popt[1], popt[0]))
            




        ax.set_ylabel("Torr")
        ax.set_yscale('log')

        return ax
    
    



    #mass is the desired mass to fit for the pressure-vs-time
    #window is a window of amu for the peak finding algorithm
    #fit is whether you want to fit a particular region (set by mintime and maxtime datetime objects)
    #duration is whether you want the primary x axis to be duration or datetimes
    def plot_mass_evolution_peakfound(self, mass, ax=None, window=5, fit=False, mintime=None, maxtime=None, duration=True, dates=True, label=None):
        if(ax is None):
            fig, ax = plt.subplots(figsize=(14,8))


        ps = []
        ts = []
        ts_h = [] #time in hours since start
        peakfound_masses = []
        for i, event in enumerate(self.data):
            p, m = self.get_peakfound_pressure(mass, i, window=window)
            if(p is None):
                continue #no data at this mass region
            ps.append(p)
            peakfound_masses.append(m)
            ts.append(event["time"])
            ts_h.append((ts[-1] - self.pump_start).total_seconds()/3600)

        if(duration and dates):
            #make an x axis on top with date labels
            axd = ax.twiny()
            l = axd.plot(ts, ps)
            l.pop().remove() #remove the line from plot
            axd.xaxis.set_major_formatter(mdates.DateFormatter("%m/%d\n %H:%M"))
            axd.grid(False)


        mstd = np.std(peakfound_masses)
        massmean = np.mean(peakfound_masses)
        if(label is None):
            label = str(round(massmean, 2)) + " amu +- " + str(round(mstd, 2))

        if(duration):
            ax.plot(ts_h, ps, '.',label=label)
            ax.set_xlabel("Hours since start")
        else:
            ax.plot(ts, ps, '.',label=label)
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
                    if(p is None):
                        continue #no data at this mass region
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
            ts_h.append((ts[-1] - self.pump_start).total_seconds()/3600)
            p, m = self.get_peakfound_pressure(mass, i, window=window)
            if(p is None):
                continue #no data at this mass region
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
    def get_peakfound_pressure(self, mass, i, window=2):
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

        #sometimes the data has no points in that mass region, in that case, skip the point
        if(len(ms_w) == 0):
            return None, None
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
        peaks, _ = find_peaks(ps_sm, distance=0.5/dm, height=5e-12)
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

        def fitfunc(x, al, A):
            return A*x**(-al)
        
        if(mintime is None or maxtime is None):
            mintime = ts[0]
            maxtime = ts[-1]
        
        ps_fit = []
        ts_fit_s = []
        ts = []
        mass_exclusion = 0.5 #amu
        for i, event in enumerate(self.data):
            data_masses = np.linspace(event["min_amu"], event["max_amu"], len(event["pressures"]))
            m_idx = (np.abs(np.array(data_masses) - mass)).argmin()
            m_returned = data_masses[m_idx]
            if(np.abs(m_returned - mass) > mass_exclusion):
                p = None
                continue #no data at this mass region
            
            p = event["pressures"][m_idx]
            ts.append(event["time"])
            if(mintime <= ts[-1] and maxtime >= ts[-1]):
                ps_fit.append(p)
                ts_fit_s.append((ts[-1] - ts[0]).total_seconds())


        guess = [1, 1e-2]
        popt, pcov = curve_fit(fitfunc, ts_fit_s, ps_fit, p0=guess)

        return fitfunc, popt, pcov, min(ts_fit_s), max(ts_fit_s)
        



    #plot time information about scans contained in the file
    def plot_scan_times(self, ax = None):
        if(ax is None):
            fig, ax = plt.subplots(figsize=(14,8))
        
        for e in self.data:
            ax.scatter(e["time"], np.max(e["pressures"]))#, color='k', s=50)
        
        return ax

    def get_time_of_scan(self, i):
        e = self.data[i]
        return(e["time"])
    
    #return the scan index closest to the input datetime object
    def get_idx_at_time(self, t):
        event = min(self.data, key=lambda x: abs((x["time"] - t).total_seconds()))
        return self.data.index(event)

    #t in hours
    def get_idx_at_duration(self, t):
        t0 = self.pump_start 
        tf = t0 + timedelta(hours=t)
        return self.get_idx_at_time(tf)


    #get total pressure by integrating the spectrum
    #for a particular scan index
    def get_total_pressure(self, i):
        e = self.data[i]
        masses = np.linspace(e["min_amu"], e["max_amu"], len(e["pressures"]))
        tot_p = np.trapz(e["pressures"], x=masses)
        return tot_p
    
    def plot_total_pressure(self, ax=None, duration=True, dates=True, label=""):
        if(ax is None):
            fig, ax = plt.subplots(figsize=(14,8))

        ps = []
        ts = []
        ts_h = [] #time in hours since start
        for i, event in enumerate(self.data):
            ts.append(event["time"])
            ts_h.append((ts[-1] - self.pump_start).total_seconds()/3600)
            ps.append(self.get_total_pressure(i))

        if(duration and dates):
            #make an x axis on top with date labels
            axd = ax.twiny()
            l = axd.plot(ts, ps)
            l.pop().remove() #remove the line from plot
            axd.xaxis.set_major_formatter(mdates.DateFormatter("%m/%d\n %H:%M"))
            axd.grid(False)


        if(duration):
            ax.plot(ts_h, ps, '.',label=label)
            ax.set_xlabel("Hours since start")
        else:
            ax.plot(ts, ps, '.',label=label)
            ax.xaxis.set_major_formatter(mdates.DateFormatter("%m/%d\n %H:%M"))

        ax.set_ylabel("Torr")
        ax.set_yscale('log')
        return ax


    #takes a single scan index and exports a file
    #that is importable to the nist RGA database tool MS Search. 
    #This file format is just a list of peak AMU's and their relative
    #magnitudes, but would not necessarily take fractional AMUs. So,
    #the user should list a set of relevant peaks, this function then 
    #does peakfinding to fit the magnitude, and organizes into the correct file format. 
    def export_for_nist(self, idx, peaks, outfile):
        ps = []
        good_peaks = []

        #if you want to check every integer amu, pass empty list
        if(peaks == []):
            e = self.data[idx]
            calib_factor = 0.5 #shift data up this many amu 
            #gaussian smooth the data 
            ms = np.linspace(e["min_amu"], e["max_amu"], len(e["pressures"]))
            ms = np.array(ms) + calib_factor
            ps_w = e["pressures"]  
            ps_w = [(_ > 0)*_ for _ in ps_w]
            ps_s = interp1d(ms, ps_w)
            int_masses = np.array(list(set(np.array(ms).astype(int))))
            good_peaks = int_masses
            ps = ps_s(int_masses)
            

        else:  
            for m in peaks:
                p, m_ret = self.get_peakfound_pressure(m, idx, window=2)
                if(p is None):
                    continue #no data at this mass region
                if(np.abs(m - m_ret) >= 1):
                    print("Couldn't get peakvalue for {:.1f} amu because of close proximity to {:.1f} amu peak".format(m, m_ret))
                else:
                    good_peaks.append(m)
                    ps.append(p)

        header = "NAME:" + datetime.strftime(self.data[idx]["time"], "%b %d, %Y  %H:%M:%S") + " idx " + str(idx) + "\n"
        header += "COMMENT: \nFORMULA:\nMW:\nCAS:\nNum Peaks: {:d}\n".format(len(good_peaks))

        #it seems the MS search program doesn't like small numbers.
        #normalize such that the max peak is 1000
        norm = np.max(ps)/1000

        datastr = ""
        for i in range(len(ps)):
            if(int(ps[i]/norm) < 1):
                continue
            datastr += "{:d} {:d};\n".format(int(good_peaks[i]), int(ps[i]/norm))

        outf = open(outfile, "w")
        outf.write(header+datastr)
        outf.close()

        return good_peaks, ps


