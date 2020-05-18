#!/usr/bin/env python
# coding: utf-8

# In[8]:


import numpy as np
from scipy import stats
import os
import matplotlib.pyplot as plt

class ExchangeRate():
    def __init__(self):
        self.dates=None # 1d array of strings for the first column of the file (omit the first element 'Series_Description')
        self.currencies=None # 1d array of strings for the first row of the file (omit the first element 'Series_Description')
        self.data=None # 2d array of float types for exchange rates from the file.
        self.num_currencies=0
        self.num_dates=0
        self.cors=None # 2d array of float types, self.cors[i,j] is the correlation coefficient between currencies i and j.
    
    def read_data(self,filename='./foreign-exchange-rates-per-dollar-20002019/Foreign_Exchange_Rates.csv'):
        """
        Read data from a csv file. 
        This function assign actual values to self.dates, self.currencies, self.data.
        Note that at this function, you need to treat the values in self.data as strings because there are missing values 'ND' in it.
        """
        
        filename=filename[-26:]             
        try:
            data=np.loadtxt(filename,dtype=str,delimiter=',')
        except FileNotFoundError:
            print('Something wrong. Please check your file name.')
        else:
            print('Data is successfully loaded.')       
        
        self.data=np.array(data[1:,1:])
        self.num_currencies=self.data.shape[1]             
        self.num_dates=self.data.shape[0]                  
        self.currencies=np.array(data[0,1:])
        self.dates=np.array(data[1:,0])
   
    def impute_missing_data(self):
        """
        Impute missing values in self.data indicated by'ND' by the value above it. 
        For example, if i>0 and self.data[i,j] is missing, self.data[i-1,j] should be used for self.data[i,j]; 
                     if i is 0 and self.data[i,j] is missing, self.data[k,j] (if available) should be used, where k >= i+1.
        After imputing all missing values, the data type of self.data should be converted from string to float.
        """
        
        for i in range(self.num_dates):
            for j in range(self.num_currencies):
                if i==0 and self.data[i,j]=='ND':   
                    self.data[i,j]=self.data[i+1,j]     
                elif self.data[i,j]=='ND':
                    self.data[i,j]=self.data[i-1,j]
        self.data=self.data.astype(np.float) 
        
    def select_columns(self,select=None):
        """
        Select some columns from self.data as indicated by the select parameter.
        INPUTS:
        select: None or 1d array of strings from self.currencies. 
                If select is None, return all columns of self.data.
                If select is a 1d arry of strings, e.g. ['UNITED_KINGDOM-POUND/US$', 'CANADA-CANADIAN$/US$', 'SWEDEN-KRONOR/US$'],
                   the corresponding columns from self.data should be returned as 2d array of shape (5217,3).
        OUTPUTS:
        data_sel: 2d array of float types, selected data.
        """
        
        if select is None:
            data_sel=self.data
        else:
            data_sel=np.zeros(shape=(self.num_dates,len(select)))
            inds=np.arange(self.num_currencies)
            for i in range(len(select)):
                ind=inds[self.currencies==select[i]]
                data_sel[:,i]=self.data[:,ind].ravel()
        return data_sel
          
    def plot(self, data_sel, select, filename='./fig_line_and_boxplot.pdf'):
        """
        Draw a line subplot, a box subplot, and a histogram subplot for data_sel in a figure,
        and save the figure to a pdf file. 
        INPUTS:
        data_sel: 2d array, the selected data from function select_columns().
        select: 1d array of string, the same meaning as in select_columns(), it is used as labels in legend.
        filename: string, the filename (with path) of the saved figure.
        """

        fig,axs=plt.subplots(3,1) # note here axs is a 1d array of axes, thus use axs[i] where i=1,2,3
        fig.set_size_inches(8,10)
        x=(np.arange(self.num_dates)).ravel()
        for i in range(len(select)): 
            axs[0].plot(x,data_sel[:,i],label=select[i])
        axs[0].set_xlabel('Date')
        axs[0].set_ylabel('Exchange Rate (per unit of USD)')
        axs[0].legend()

        data=[]
        for i in range(len(select)):
            data.append(data_sel[:,i])
        axs[1].boxplot(data)
        axs[1].set_xticklabels(select)
        axs[1].set_xlabel('Currency')
        axs[1].set_ylabel('Exchange Rate (per unit of USD)')
        
        for i in range(len(select)):
            fc=(np.random.random(),np.random.random(),np.random.random(),0.5)
            axs[2].hist(data_sel[:,i],bins=150,fc=fc,label=select[i])
        axs[2].set_xlabel('Exchange Rate (per unit of USD)')
        axs[2].set_ylabel('Density')
        axs[2].legend()
        
        try:
            plt.savefig(filename)
        except FileNotFoundError:
            print('Something wrong. Please check your file name')
        else:
            print('File saved in {0}'.format(filename))
            
    def compute_cor(self):
        """
        Compute the Pearson correlation coefficients for any pairs of currencies. Save the result in the 2d array self.cors.
        self.cors[i,j] is the Pearson correlation coefficient for currency i (the i-th column of self.data) and currency j (the j-th column of self.data). 
        """
        
        self.cors=np.zeros( (self.num_currencies,self.num_currencies) )
        n=-1
        for c in range(self.num_currencies):
            n=n+1
            x=self.data[:,c]
            i=0
            for c in range(self.num_currencies):
                y=self.data[:,c]
                coeff,pval=stats.pearsonr(x,y)
                self.cors[n][i]=coeff
                i=i+1

    def plot_heatmap(self, filename='./fig_heatmap_cor.pdf'):
        """
        Draw heatmap of the correlation coefficients in self.cors, and save it to a pdf file. 
        """
        
        fig,ax=plt.subplots(1,1)
        fig.set_size_inches(6,6)
        im = ax.imshow(self.cors)
        ax.set_xticks(np.arange(self.num_currencies))
        ax.set_yticks(np.arange(self.num_currencies))
        ax.set_xticklabels(self.currencies)
        ax.set_yticklabels(self.currencies)
        plt.setp(ax.get_xticklabels(), rotation=90)
        plt.colorbar(im)
        
        try:
            plt.savefig(filename)
        except FileNotFoundError:
            print('Something wrong. Please check your filename.')
        else:
            print('File saved in {0}'.format(filename))
            
    def search(self, currency, date):
        """
        Search for the exchange rate of a currency on a specifiec date.
        INPUTS:
        currency: string, must be one element from self.currencies.
        date: string, must be one element from self.dates.
        OUTPUTS:
        exrate: scalar of float type, for the desired exchange rate.
        """
        
        inds=np.arange(self.num_currencies)
        ind=inds[self.currencies==currency] 
        inds=np.arange(self.num_dates)
        index=inds[self.dates==date]
        return self.data[index,ind][0]
    


# In[9]:


# Create an instance for this class, and read data
er=ExchangeRate()
er.read_data()


# In[10]:


# Impute missing data
er.impute_missing_data()
er.data


# In[11]:


# Obtain data for AUD, CAD, and SGD, and draw line plot, box-plot, & histogram
select=np.array(['AUSTRALIA-AU$/US$', 'CANADA-CANADIAN$/US$', 'SINGAPORE-SINGAPORE$/US$'])
data_sel=er.select_columns(select) 
er.plot(data_sel, select, filename='./fig_line_and_boxplot.pdf')


# In[12]:


# Obtain correlation coefficients between any pairs of currencies, and plot it in heatmap
er.compute_cor()
er.plot_heatmap(filename='./fig_heatmap_cor.pdf')


# In[13]:


# Search for the exchange rate of a currency on a particular day\
cur='UNITED_KINGDOM-POUND/US$'
da='2019-12-25'
exrate=er.search( cur, da ) 
print('Exchange rate is {0} for {1} on {2}'.format(exrate, cur, da))

