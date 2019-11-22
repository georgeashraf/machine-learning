import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from matplotlib import colors
import matplotlib as mpl
from scipy import linalg
def dataset_cov():
    '''Generate 2 Gaussians samples with different covariance matrices'''
    n, dim = 300, 2
    np.random.seed(0)
    C = np.array([[0., -1.], [2.5, .7]]) * 2.
    X = np.r_[np.dot(np.random.randn(n, dim), C),
              np.dot(np.random.randn(n, dim), C.T) + np.array([1, 4])]
    y = np.hstack((np.zeros(n), np.ones(n)))
    return X, y
cmap = colors.LinearSegmentedColormap(
    'red_blue_classes',
    {'red': [(0, 1, 1), (1, 0.7, 0.7)],
     'green': [(0, 0.7, 0.7), (1, 0.7, 0.7)],
     'blue': [(0, 0.7, 0.7), (1, 1, 1)]})
plt.cm.register_cmap(cmap=cmap)

def calculate_phi(Y):
    m = len(Y)
    return sum(y == 1 for y in Y)/m

def calculate_mu1(X, Y):
    pos=sum(y == 1 for y in Y)
    ans=np.dot(Y.transpose(), X).transpose()
    mu1=ans/pos
    return mu1

def calculate_mu0(X, Y):
    neg=sum(y == 0 for y in Y)
    Y_temp = 1.0 - Y
    ans=np.dot( Y_temp .transpose(), X).transpose()
    mu0=ans/neg
    return mu0

def calculate_sigma0(mu0,X,Y):
     neg = X[Y == 0] - mu0
     sigma = ((neg.T).dot(neg) )/len(Y)
     return sigma
 
def calculate_sigma1(mu1,X,Y):    
     pos = X[Y == 1] - mu1
     sigma = ((pos.T).dot(pos) )/len(Y)
     return sigma
def calculate_prob_xgiveny(x,mu,sigma) :
  pi = 3.14
  dim = len(mu)
  inv_sigma=np.linalg.inv(sigma)
  return np.exp(-0.5 * (x - mu).T @ (inv_sigma @ (x - mu))) / (2 * pi)**(dim/2) * np.sqrt(np.linalg.det(sigma))


def calculate_prob_y(y,phi):
    if y==1:
        return phi
    else:
        return 1-phi
    
def train(X,Y):
      phi=calculate_phi(Y)
      mu1=calculate_mu1(X, Y)
      mu0=calculate_mu0(X, Y)
      sigma0=calculate_sigma0(mu0,X,Y)
      sigma1=calculate_sigma1(mu1,X,Y)
      #print(mu0,mu1,sigma)
      return phi,mu0,mu1,sigma0,sigma1
  
    
def predict(x,phi,mu0,mu1,sigma0,sigma1):
       px_0 = calculate_prob_xgiveny(x, mu0, sigma0)*calculate_prob_y(0, phi)
       px_1 = calculate_prob_xgiveny(x, mu1, sigma1)*calculate_prob_y(1, phi)
       if px_0>px_1:
         return 0  
       else:
           return 1
       
def predict_labels(X,phi,mu0,mu1,sigma0,sigma1):
   labels=[]
   for i in X:
       px_0 = calculate_prob_xgiveny(i, mu0, sigma0)*calculate_prob_y(0, phi)
       px_1 = calculate_prob_xgiveny(i, mu1, sigma1)*calculate_prob_y(1, phi)
       if px_0>px_1:
         labels.append( 0. ) 
       else:
           labels.append( 1. ) 
   return labels        
def plot_ellipse(splot, mean, cov, color):
    v, w = linalg.eigh(cov)
    u = w[0] / linalg.norm(w[0])
    angle = np.arctan(u[1] / u[0])
    angle = 180 * angle / np.pi  # convert to degrees
    # filled Gaussian at 2 standard deviation
    ell = mpl.patches.Ellipse(mean, 2 * v[0] ** 0.5, 2 * v[1] ** 0.5,
                              180 + angle, facecolor=color,
                              edgecolor='black', linewidth=2)
    ell.set_clip_box(splot.bbox)
    ell.set_alpha(0.2)
    splot.add_artist(ell)
    
def plot_qda_cov(mu0,mu1,sigma0,sigma1 ,splot):
    plot_ellipse(splot, mu0, sigma0, 'red')
    plot_ellipse(splot, mu1, sigma1, 'blue')

def result_plot(X, y,phi,mu0,mu1,sigma0,sigma1):
    fig, splot = plt.subplots(1,1,figsize=(10,10))
    plt.title('Quadratic Discriminant Analysis')
    plt.ylabel('Data with\n different covariance')
    X_set = X    
    y_pred=predict_labels(X_set,phi,mu0,mu1,sigma0,sigma1)
    tp = (y == y_pred)  # True Positive
    tp0, tp1 = tp[y == 0], tp[y == 1]
    XX0, XX1 = X[y == 0], X[y == 1]
    X0_tp, X0_fp = XX0[tp0], XX0[~tp0]
    X1_tp, X1_fp = XX1[tp1], XX1[~tp1]
    # class 0: dots
    plt.scatter(X0_tp[:, 0], X0_tp[:, 1], marker='.', color='red' ,label='0')
    plt.scatter(X0_fp[:, 0], X0_fp[:, 1], marker='x',
                s=20, color='#990000')  # dark red

    # class 1: dots
    plt.scatter(X1_tp[:, 0], X1_tp[:, 1], marker='.', color='blue',label='1')
    plt.scatter(X1_fp[:, 0], X1_fp[:, 1], marker='x',
                s=20, color='#000099')  # dark blue
                
    nx, ny = 400, 300
    x_min, x_max = plt.xlim()
    y_min, y_max = plt.ylim()    
    X1, X2 =np.meshgrid(np.linspace(x_min, x_max, nx),
                         np.linspace(y_min, y_max, ny)) 
    contour_pred = predict_labels(np.array([X1.ravel(), X2.ravel()]).T,phi,mu0,mu1,sigma0,sigma1)                
    contour_pred=np.array(contour_pred)  .reshape(X1.shape)   
    plt.pcolormesh(X1, X2, contour_pred, cmap='red_blue_classes',
                   norm=colors.Normalize(0., 1.), zorder=0)
    plt.contour(X1, X2, contour_pred,[0.5], linewidths=1., colors='white')
    plt.xlim(X1.min(), X1.max())
    plt.ylim(X2.min(), X2.max())
    plt.plot(mu0[0], mu0[1],
             '*', color='yellow', markersize=15, markeredgecolor='grey')
    plt.plot(mu1[0], mu1[1],
             '*', color='yellow', markersize=15, markeredgecolor='grey')    
    plt.legend()
    return splot       
if __name__ == '__main__':
   iris = datasets.load_iris()
   idx = iris.target != 2
   data = iris.data[idx].astype(np.float32)
   Y = iris.target[idx].astype(np.int32)
   X = data[:, :2]  # consider only 2 features
   phi,mu0,mu1,sigma0,sigma1=train(X,Y)
   splot=result_plot(X, Y,phi,mu0,mu1,sigma0,sigma1)
   plot_qda_cov(mu0,mu1,sigma0,sigma1, splot)
   plt.savefig('QDA.png')
   plt.show()

  
  
  