# -*- coding: utf-8 -*-
"""
Created on Wed Jun  17 12:47:42 2019
@author: david.searle
"""
import random
import numpy as np
import pandas as pd

class MarkovAttribution():
    def __init__(self, 
                 path_prefix='T_',
                 conversion_col='conv_flag',
                 removal_calc_mode='approximate',
                 removal_leftover_redist='null',
                 synthesize_n=20000):
        """Initialize the attribution module.
        
        Args:
            df (DataFrame): Pandas DataFrame containing the user journeys.
            path_prefix (str): Prefix for all 'touchpoint' columns.
            conversion_col (str): Column containing the conversion status
            removal_calc_mode (str): Valid options ('synthesize','approximate')
            removal_leftover_redist (str): Valid options ('even','null') 
            synthesize_n (int): Number of paths to synthesize if using synth
        Returns:
            
        """
        self.path_prefix = path_prefix
        self.conversion_col = conversion_col
        self.removal_calc_mode = removal_calc_mode
        self.synthesize_n = synthesize_n
        self.removal_leftover_redist = removal_leftover_redist
        
    def append_paths(self,df):
        """Convert raw pandas DataFrame to modelling DataFrame. 
            Appends new column with touchpoints split by > seperator.
        
        Args:
            df (DataFrame): Raw DataFrame containing user journeys.
        
        Returns:
            p (DataFrame): Formatted DataFrame with complete Paths column.
                
        """
        touch_cols = [col for col in df.columns if self.path_prefix in col] + \
                     [self.conversion_col]
        df[self.conversion_col] = df[self.conversion_col].fillna('null')  
        df['Paths'] = [' > '.join([str(x) for x in r if str(x) != 'nan'])\
                          for r in df[touch_cols].iloc[:,:].values.tolist()]
        df['Paths'] = df['Paths'].map(lambda x: 'start > '+x)
        
        return df
    
    def fit(self, df):
        """Runs the markov modelling attribution process.
        
        Args:
        
        Returns:
            attribution (object): Object containing - Touchpoint Totals
                                                    - Transition Probabilities
                                                    - Transition Matrix
                                                    - Removal Effects
                                                    - Markov Values
                
        """
        df = self.append_paths(df)
        
        self.total_conversions = sum(df.Paths.str.count(" > conv"))
        self.base_conv_rate = sum(df.Paths.str.count(" > conv")) / len(df)
        self.paths = [i for i in list(df.Paths.str.split(' > ').values)]
        self.n_paths = len(df)
        self.touchpoints = set(x for el in self.paths for x in el)
        
        transition_details = self.conversion_states(df.Paths)
        tran_states = transition_details['Transition States']
        conv_counts = transition_details['Transition Conversion Counts']
        self.tran_prob = self.transition_probabilities(tran_states=tran_states, 
                                                       conv_counts=conv_counts)
        self.tran_matrix = self.gen_transition_matrix(self.tran_prob)
        self.removal_effects = self.calculate_removal_effect(self.tran_matrix)
        self.markov_values =self.markov_attributed_values(self.removal_effects)
        
        return {'Touchpoint Totals': conv_counts,
                'Transition Probabilities': self.tran_prob,
                'Transition Matrix': self.tran_matrix,
                'Removal Effects':self.removal_effects,
                'Markov Values':self.markov_values}
        
    def conversion_states(self, paths):
        """ Calculate counts of transitions
            Calculate how often a state is contained in a converting path
        
        Args:
        
        Returns:
            transition (dict): Dict contains - Transition States
                                             - Transition Conversion Counts
                
        """
        tran_states = {}
        conv_count = {}
        for x in self.touchpoints:
            conv_count[x] = 0
            for y in self.touchpoints:
                tran_states[x+" > "+y] = paths.str.contains(x+" > "+y).sum()
                if x not in ['conv','null']:
                    conv_count[x] += paths.str.contains(x+" > "+y).sum()
        return {'Transition States': tran_states,
                'Transition Conversion Counts': conv_count}

    def transition_probabilities(self, tran_states, conv_counts):
        """ Calculate the probability of a transition between touchpoints in
            long format
        
        Args:
        
        Returns:
            tt (DataFrame): From - 'From' Channel
                            To - 'To' Channel
                            Prob - Represents p(From > To)
                            Inner prob - Represents p(From > To | From).
                
        """
        ts = pd.DataFrame(tran_states.items(),columns=['tran','tcount'])
        tt = ts['tran'].str.split(' > ', n=2, expand=True).join(ts[['tcount']])
        tt.columns = ['from','to', 'tcount']
        tt['from'] = tt['from'].str.strip()
        tt['to'] = tt['to'].str.strip()
        tt['prob'] = tt['tcount'] / tt['tcount'].sum()
        tt['inner_prob'] = tt.tcount/tt.groupby('from').tcount.transform('sum')  
        tt.fillna(0, inplace=True)
        return tt
    
    def gen_transition_matrix(self, tran_prob):
        """ Calculate the probability of a transition between touchpoints in
            wide format (n_unique_touch x n_unique_touch)
        
        Args:
            tran_prob (DataFrame): Dataframe containing long format transitions
            
        Returns:
            tran_matrix (DataFrame): Column names will match index/touchpoints
                
        """
        tran_matrix=tran_prob.pivot(index='from',columns='to')[['inner_prob']]
        tran_matrix.columns = [f[1] for f in tran_matrix.columns]
        tran_matrix = tran_matrix.reindex(sorted(tran_matrix.columns), axis=1)
        tran_matrix.sort_index(inplace=True)
        tran_matrix.fillna(0, inplace=True)
        tran_matrix.loc['conv']['conv'] = 1.0
        tran_matrix.loc['null']['null'] = 1.0
        
        return tran_matrix
    
    def generate_removal_transition(self, df, channel_name):
        """ Genereates new transition matrix without the removal channel.
            Redistributes transition likelihood.
        
        Args:
            df (DataFrame): Full Transition Matrix
            channel_name (str): Channel to remove
        Returns:
            removal (DataFrame): New transition matrix
                
        """
        rdf = df.copy(deep=True)
        removal = rdf.drop(channel_name, axis=1).drop(channel_name, axis=0)
        
        for col in removal.columns:
            leftover = 1.0 - np.sum(list(removal.loc[col]))
            if leftover != 0:
                if self.removal_leftover_redist == 'even':
                    removal.loc[col] =removal.loc[col] / removal.loc[col].sum()
                else:
                    removal.loc[col]['null'] = leftover
                    
        removal.loc['null']['null'] = 1.0
        
        return removal
        
    def generate_synthetic_pathways(self, transitions, n_journeys):
        """ Synthesizes user journeys given a transition matrix
        
        Args:
            transitions (DataFrame): Transition matrix (m x m)
            n_journeys (int): Number of user journeys to create
            
        Returns:
            journeys (array): List of created journeys
            conversions (int): Number of journeys that results in conversion
                
        """
        conversions = 0
        max_pathway = 20
        journeys = []
        for journey in range(n_journeys):
            jour= ['start']
            m = 1
            end_journey = False
            next_touch_dist = transitions.loc['start']
            while m <= max_pathway and end_journey==False:
                touch_list = list(next_touch_dist.index.values)
                touch_dist = next_touch_dist.values
                next_touch =  random.choices(touch_list, touch_dist)[0]
                if next_touch=='conv':
                    conversions += 1
                if m == max_pathway:
                    next_touch = 'null'
                jour.append(next_touch)
                if next_touch=='conv' or next_touch=='null':
                    end_journey= True
                next_touch_dist = transitions.loc[next_touch]
                m+=1
            journeys.append(' > '.join(jour))
            
        return journeys, conversions
    
    def removal_effect_synth(self, df, channel_name):
        """ Calculates removal effects via synthesis of journeys then counting
        resulting journeys that ended in success. Slower than approx version.
        
        Args:
            df (DataFrame): Transition matrix (m x m)
            channel_name (string): Channel to remove and calc removal effect on
            
        Returns:
            removal_effect (float): Removal effect of the channel specified
                
        """
        removal = self.generate_removal_transition(df, channel_name)
        _, conversions = self.generate_synthetic_pathways(transitions=removal,
                                                  n_journeys=self.synthesize_n)
        removal_conv = conversions / self.synthesize_n
        removal_effect = self.removal_effect_score(removal_conv)
        return removal_effect
    
    def removal_effect_score(self, removal_conversion_rate):
        """ Calculates removal effect
            When the conversion rate is HIGHER without the removed channel than
            the baseline conversion rate, this could lead to a negative score
            as we're saying performance has improved without that channel
        Args:
            removal_conversion_rate (float): Conv rate for removal channel
            
        Returns:
            removal_effect (float): Calculated removal effect
                
        """
        return 1 - (removal_conversion_rate / self.base_conv_rate)
        
    def removal_effect_approx(self, df, channel_name):
        """ Calculates removal effects via direct approximation/linear algebra
            ~Much faster than synthesis version~
        Args:
            df (DataFrame): Transition matrix (m x m)
            channel_name (string): Channel to remove and calc removal effect on
            
        Returns:
            removal_effect (float): Removal effect of the channel specified
                
        """
        removal = self.generate_removal_transition(df, channel_name)
        #Transition probabilities A>B (excluding final transition)
        tp = removal.drop(['null','conv'],axis=1).drop(['null', 'conv'],axis=0)
        #Final transition probabilities -without the channel thats been removed
        fp = removal[['null', 'conv']].drop(['null', 'conv'], axis=0)
        #Approximation of end result -- (I−A)^−1 represents a power series tr
        traversed = np.linalg.inv(np.identity(np.shape(tp)[0])-np.asarray(tp))
        #Finally, assume weve traversed a path and ended up on some touchpoint
        #Caclulate the final step (either conv or non-conv)
        outcome = pd.DataFrame(np.dot(traversed,np.asarray(fp)),index=fp.index)
        #Assume we're at the starting position -> find the chance of 
        #conversion given the removed channel doesn't exist
        removal_conv = outcome[[1]].loc['start'].values[0]
        removal_effect = self.removal_effect_score(removal_conv)
        return removal_effect
        
    def calculate_removal_effect(self, df_transition):
        """ For each of the channels, computes the effect of removing that
            channel from the user journey.
            
        Args:
            df_transition (DataFrame): Transition matrix (m x m)
            
        Returns:
            effects (dict): Removal effects of each channel
                
        """
        touch = [i for i in self.touchpoints \
                                         if i not in ['conv', 'null', 'start']]
        effects = {}
        for channel in touch:
            if self.removal_calc_mode == 'synthesize':
                r_effect = self.removal_effect_synth(df_transition, channel)
            else:
                r_effect = self.removal_effect_approx(df_transition, channel)
                
            effects[channel] = r_effect
        
        return effects
        
    def markov_attributed_values(self, removal_effects):
        """ Computes markov attributed values.
          The larger the removal effect, the larger the markov attributed value
            
        Args:
            removal_effects (dict): Dictionary of {channel:removal effect}
            
        Returns:
            markov_conversions (dict): Markov attributed values of each channel
                
        """
        min_removal = np.min(list(removal_effects.values()))
        
        if min_removal < 1:
            for k, v in removal_effects.items():
                removal_effects[k] = v + abs(min_removal) + 0.1
            
        denom = np.sum(list(removal_effects.values()))
        
        allocation_amt = list()
        for i in removal_effects.values():
            allocation_amt.append((i / denom) * self.total_conversions)
        
        markov_conversions = dict()
        i = 0
        for channel in removal_effects.keys():
            markov_conversions[channel] = allocation_amt[i]
            i += 1
            
        return markov_conversions