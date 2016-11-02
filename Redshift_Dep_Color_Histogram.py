## Create the pretty colored histogram that illustrates evolution with redshift

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cmx
import matplotlib.colors as colors
from time import strftime

def multicolor_hist( S, z, xmin, xmax, dM, bin_num = 50, save_fig = False, **kwargs):

    cmap_color = kwargs.pop("color", "gist_rainbow_r")
    xlabel_fontsize = kwargs.pop("xlabel_size", 40)
    xticks_fontsize = kwargs.pop("xlabel_tick", 30)
    yticks_fontsize = kwargs.pop("xlabel_tick", 20)
    legend_fontsize = kwargs.pop("legend_font", 20)
    figure_size = kwargs.pop('fig_size', (18.0, 12.0))
    
    for i in range(0, len(z)):
        z[i]=round(z[i], 2)
    mylegend = map(str, z)

    bins = np.linspace(xmin, xmax, (bin_num + 1)) # use odd number of bins
    bincenters = 0.5*(bins[1:]+bins[:-1])

    color_option  = plt.get_cmap(cmap_color)
    cNorm = colors.Normalize(z.min(), z.max())
    scalarMap = cmx.ScalarMappable(norm=cNorm, cmap=color_option)

    n  = [[] for i in range(len(z))]#np.zeros(len(bins))
    bins1  = [[] for i in range(len(z))]
    for i in range(0, len(z)):
        n[i], bins1[i], patches =plt.hist(S[i], bins, stacked = True, normed = True, label=mylegend[i], color= scalarMap.to_rgba(z[i]), alpha=0.5)
    plt.close()

    fig=plt.figure(figsize=figure_size) ### make this a kwarg or something ###

    size_bins= (bins[1]-bins[0])

    colList =[]
    bottoms2=[[ 0 for i in range(len(n[0])) ] for j in range(len(n[0]))]
    for j in range(0,len(n[0])):
        col = j
        colList=[]
        for i in range(0,len(n)):
            colList +=[n[i][col]]
        bottoms2[j] = colList
    bottoms3=[[ 0 for i in range(len(z)) ] for j in range(len(n[0]))] # column first then row

    for j in range(1, len(z)):
        for i in range (0, len(n[0])):
            bottoms3[i][j] = np.cumsum(bottoms2[i])[j-1]
            bottoms3[i][0]=0.
    #        
    bottoms4 = zip(*bottoms3)

    for i in range(0, len(n)):
        plt.bar(bincenters, n[i],size_bins, color =scalarMap.to_rgba(z[i]), label=mylegend[i], alpha = 0.5,align = 'center', bottom=bottoms4[i] )


    plt.legend(bbox_to_anchor = (1.0,1), loc=2, borderaxespad=0., fontsize=legend_fontsize, title = 'Redshift')
    plt.tick_params(axis='x', labelsize=xticks_fontsize)
    plt.tick_params(axis='y', labelsize=yticks_fontsize)
    plt.xlabel('M [mag]', fontsize = xlabel_fontsize)
    plt.ylabel('# of SNe', fontsize = xlabel_fontsize)
    if save_fig:
        plt.savefig('../Figures/%s_Multiple_redshift_Hist_%s_SN_%.0f.pdf'%(strftime('%Y%m%d'), len(S[0])*len(z), dM*100))
    #plt.show()
    return fig


def subplot(S, s, z, xmin, xmax, dM, bin_num = 50, save_fig = False, **kwargs):#fig1, fig2, dim):
    
    cmap_color = kwargs.pop("color", "gist_rainbow")
    xlabel_fontsize = kwargs.pop("xlabel_size", 40)
    xticks_fontsize = kwargs.pop("xlabel_tick", 30)
    yticks_fontsize = kwargs.pop("xlabel_tick", 20)
    legend_fontsize = kwargs.pop("legend_font", 20)
    figure_size = kwargs.pop('fig_size', (18.0, 12.0))

    fig, (ax1,ax2) = plt.subplots(1, 2, figsize=figure_size, sharey = True)
    fig.subplots_adjust(left=0.075, wspace=0.0) #left=0.2,
    #fig = plt.figure(figsize=(10,10))
    #rect = [0.1,0.2,0.6,0.6]
    #ax1 = fig.add_axes(rect)
    #rect2 = [0.65001,0.2,0.6,0.6]#[0.55,0.12,0.4,0.4]
    #ax2 = fig.add_axes(rect2)
    
    for i in range(0, len(z)):
        z[i]=round(z[i], 2)
    mylegend = map(str, z)

    bins = np.linspace(xmin, xmax, (bin_num + 1)) # use odd number of bins
    bincenters = 0.5*(bins[1:]+bins[:-1])

    color_option  = plt.get_cmap(cmap_color)
    cNorm = colors.Normalize(z.min(), z.max())
    scalarMap = cmx.ScalarMappable(norm=cNorm, cmap=color_option)

    n  = [[] for i in range(len(z))]#np.zeros(len(bins))
    bins1  = [[] for i in range(len(z))]
    for i in range(0, len(z)):
        n[i], bins1[i], patches =plt.hist(S[i], bins, stacked = True, normed = True, label=mylegend[i], color= scalarMap.to_rgba(z[i]), alpha=0.5)
    plt.close()

    n_1  = [[] for i in range(len(z))]#np.zeros(len(bins))
    bins1_1  = [[] for i in range(len(z))]
    for i in range(0, len(z)):
        n_1[i], bins1_1[i], patches_1 =plt.hist(s[i], bins, stacked = True, normed = True, label=mylegend[i], color= scalarMap.to_rgba(z[i]), alpha=0.5)
    plt.close()

    size_bins= (bins[1]-bins[0])

    colList =[]
    bottoms2=[[ 0 for i in range(len(n[0])) ] for j in range(len(n[0]))]
    for j in range(0,len(n[0])):
        col = j
        colList=[]
        for i in range(0,len(n)):
            colList +=[n[i][col]]
        bottoms2[j] = colList
    bottoms3=[[ 0 for i in range(len(z)) ] for j in range(len(n[0]))] # column first then row

    for j in range(1, len(z)):
        for i in range (0, len(n[0])):
            bottoms3[i][j] = np.cumsum(bottoms2[i])[j-1]
            bottoms3[i][0]=0.
    #        
    bottoms4 = zip(*bottoms3)

    for i in range(0, len(n)):
        ax1.bar(bincenters, n[i],size_bins, color =scalarMap.to_rgba(z[i]), alpha = 0.5,align = 'center', bottom=bottoms4[i] ) # label=mylegend[i]

    colList_1 =[]
    bottoms2_1=[[ 0 for i in range(len(n_1[0])) ] for j in range(len(n_1[0]))]
    for j in range(0,len(n_1[0])):
        col_1 = j
        colList_1=[]
        for i in range(0,len(n_1)):
            colList_1 +=[n_1[i][col_1]]
        bottoms2_1[j] = colList_1
    bottoms3_1=[[ 0 for i in range(len(z)) ] for j in range(len(n_1[0]))] # column first then row

    for j in range(1, len(z)):
        for i in range (0, len(n_1[0])):
            bottoms3_1[i][j] = np.cumsum(bottoms2_1[i])[j-1]
            bottoms3_1[i][0]=0.
    #        
    bottoms4_1 = zip(*bottoms3_1)
    ax2.clear()
    for i in range(0, len(n_1)):
        ax2.bar(bincenters, n_1[i],size_bins, color =scalarMap.to_rgba(z[i]), label=mylegend[i], alpha = 0.5,align = 'center', bottom=bottoms4_1[i] )
    ax2.yaxis.set_visible(False)

    ax1.set_xlim(-20.3, -18.7)
    ax2.set_xlim(-20.3, -18.7)

    l = ax2.legend(bbox_to_anchor = (1.0,1), loc=2, borderaxespad=0., fontsize=legend_fontsize, title = 'Redshift') #legend =
    plt.setp(l.get_title(),fontsize=legend_fontsize)
    
    ax1.tick_params(axis='x', labelsize=xticks_fontsize)
    ax1.set_xticks([-20.0, -19.5, -19.0])
    ax2.set_xticks([-20.0, -19.5, -19.0])
    ax2.tick_params(axis='x', labelsize=xticks_fontsize)
    ax1.tick_params(axis='y', labelsize=yticks_fontsize)
    ax1.set_title('$\Delta M = 0.05$ mag', fontsize = xlabel_fontsize)
    ax2.set_title('$\Delta M = 0.5$ mag', fontsize = xlabel_fontsize)
    fig.suptitle('$M$ [mag]', fontsize = xlabel_fontsize, x = 0.5, y = 0.075)
    ax1.set_ylabel('# of SNe', fontsize = xlabel_fontsize)

    if save_fig:
        fig.savefig('../Figures/%s_Multiple_redshift_Hist_10000_SN.pdf'%(strftime('%Y%m%d')))
    #plt.show()
    return fig
