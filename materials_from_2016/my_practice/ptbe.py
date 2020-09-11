# ptbe.py
"""
A collection of functions that serve to extend PythTB capabilities
"""

def print_eig_real(eval,evec):
    """
    Prints eigenvalues and real parts of eigenvectors, one to a line.
    Should not be used if eigenvectors are complex.  Also, there is
    no line wrapping, so this should not be used for long eigenvectors.
    """
    n=len(eval)
    evecr=evec.real
    print "  n   eigval   eigvec"
    for i in range(n):
        print " %2i  %7.3f" % (i,eval[i]),
        print "  ("+", ".join("%6.2f" % x for x in evecr[i,:])+" )"

def plot_bsr(plot_file,plot_title,k_vec,k_dist,k_node,k_lab,evals):
    """
    Automates the steps needed to plot a bandstructure
    """
    # import pyplot from matplotlib
    import matplotlib.pyplot as pl
    # first make a figure object
    fig=pl.figure()
    # put title
    pl.title(plot_title)
    # specify horizontal axis details
    a = pl.gca()
    # set range of horizontal axis
    a.set_xlim([0,k_node[-1]])
    # put tickmarks and labels at node positions
    pl.xticks(k_node,k_lab)
    # add vertical lines at node positions
    for n in range(len(k_node)):
        a.axvline(x=k_node[n],linewidth=1, color='k')
    # plot bands
    for n in range(evals.shape[0]):
        pl.plot(k_dist,evals[n])
    # make an PDF figure of a plot
    pl.savefig(plot_file)

    
