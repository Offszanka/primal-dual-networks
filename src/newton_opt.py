# -*- coding: utf-8 -*-
# from warnings import catch_warnings, simplefilter, warn

import matplotlib.pyplot as plt
import numpy as np
import tqdm
import scipy

def armijo_line_search(func, grad_f, xk, dk,*,
                       beta=0.5, sigma=1e-4, max_iter=10,
                       fxk = None, func_params = None):
    """    Performs Armijo line search backtracking.

    Args:
        func (callable): The Function.
        grad_f (ndarray): The gradient of func at xk.
        xk (ndarray): Start point.
        dk (ndarray): Seach direction.
        beta (float, optional): Step size. Defaults to 0.5.
        sigma (float, optional): derivative float constant. The default is 1e-4.
        max_iter (int, optional): maximum number of iterations to perform. The default is 10.
        fxk (_type_, optional): _description_. Defaults to None.
        func_params (float, optional): function value at xk. This is calculated if it is omitted.

    Returns:
        Found step size
        number of iterations
    """
    max_iter = int(max_iter)
    alpha = 1.
    dot_product = np.dot(grad_f.ravel(), dk.ravel())

    if func_params is None:
        func_params = dict()

    if fxk is None:
        fxk = func(xk, **func_params)

    for i in range(max_iter):
        if func(xk + alpha*dk, **func_params) >= fxk + sigma*alpha*dot_product:
            alpha *= beta
        else:
            return alpha, i

    return alpha, i

class Newton:
    """Represents a newton iteration. Performs the classical Newton iteration with Armijo line backtracking.
    Is callable.
    From Ulbrich&Ulbrich (2012), Nichtlineare Optimierung
    """

    def __init__(self, func, fprime, hess_f,*,
                 documentation = False, **params):
        """
        Parameters
        ----------
        func : TYPE
            The function to minimize.
        fprime : function
            A method to provide the derivative of f at a point x0.
        hess_f : function
            A method to provide the hessian of f at a point x0.
        solver : TYPE, optional
            Der Gleichungssystemrechner, der das System Hess(f)xk = -dk löst.
            Muss als erstes Argument die Hessematrix nehmen und als zweites Argument die Abstiegsrichtung.
            Die müssen natürlich zueinander kompatibel sein.
            Muss als output die Lösung und eine info haben.
            (So wie gmres von scipy)
            The default is scipy.sparse.gmres.

        Returns
        -------
        None.

        """
        self.func = func
        self.getDerivative = fprime
        self.getHessian = hess_f

        self.documentation = documentation
        self.doc = None
        self.history = None

        self.N = 0 # To save the number of iterations performed in a call
        self.total_N = 0 # To save the number of iterations performed by this object.
        self.params = dict(
            t1=1e-6,
            t2=1e-6,
            p=0.1,
            sigma=1e-4,
            beta=0.5,
            max_iter=1e3,
            tol=1e-3,
            armijo_N=10,
            sparse_hessian = False,
            armijo_max_iter = 10,
            solver = None,
            )
        self.update_params(**params)


    def update_params(self, inplace = True, **params):
        """
        To change the parameters globally for this whole object.

        Parameters
        ----------
        inplace : bool, optional
            Indicates if the parameters of this object should be changed (permanently) or only change one time.
            If True then the parameters are overwritten and the parameters are returned.
            If False then the parameters are not overwritten and a copy of the updated parameters is returned.
            The default is True.
        t1 : TYPE, optional
            Gradientenbedingung links im Minimum. The default is 1e-6.
        t2 : TYPE, optional
            Gradientenbedingung rechts im Minimum . The default is 1e-6.
        p : TYPE, optional
            DESCRIPTION. The default is 0.1.
        sigma : TYPE, optional
            DESCRIPTION. The default is 1e-4.
        beta : TYPE, optional
            Armijo step size. The default is 0.5.
        armijo_N : int, optional
            Number of armijo line search iteration. The default is 10.
        max_iter : int, optional
            maximum number of iterations. The default is 1e3.
        tol : double, optional
            tolerance. The gradient should be smaller then this. The default is 1e-3.
        solver : callable, optional
            The solver method to solve the system hessian_f sk = -grad_f for sk. The signature should be
            sk = solver(hessian_f, -grad_f) and everything should be already given to solver as for example the tolerance.
        sparse_hessian : bool, optional
            Indicates if the hessian is a sparse matrix. If sparse_hessian == True then spsolve
            from scipy.sparse.linalg is used to determine the solution.
            Otherwise np.linalg.solve is used.
            Caution is advised as np.linalg.solve can take a lot of memory space (and computational time) for big hessian matrices.
            Only necessary if solver is not specified.

        Raises
        ------
        ValueError if a parameter is not correctly passed.

        Returns
        -------
        None.

        """
        p = set(params.keys())
        sp = set(self.params.keys())
        if not p.issubset(sp):
            raise ValueError(f'{p - sp} are not valid parameters. Valid parameters are {sp}')
        else:
            if inplace:
                self.params.update(params)
                return self.params.copy()
            else:
                sparams = self.params.copy()
                sparams.update(params)
                return sparams


    def find_min(self, x_0 : np.ndarray, *,
                 verbose : bool = False, documentation : bool = None, save_xk : bool = False,
                 **params):
        """
        The (global) newton algorithm with Armijo line search.

        Parameters
        ----------
        x_0 : ndarray
            starting point.
        verbose : boolean, optional
            determine if the loop should be shown in the console
        documentation : boolean
            determine if values inside the loop should be saved. They can be accessed through the attribute history.
            Can be None to use the class attribute documentation which was saved when initializing this object.
        save_xk : boolean, optional
            determine if xk should be saved. This can be dangerous
            as it might use a lot of memory space.
        solver : function, optional
            A custom solver to solve the equation
            sk = solver(hessian_f, -grad_f)
            You set up everything to ensure that the function can be evaluated
            and gives the right result.
        params, optional
            additional parameters for the newton iteration.
            See "update_params" for more information.
        Returns
        -------
        xk : ndarray
            Result, which is the minimizing argument.

        """
        ## Preparing the parameters
        sparams = self.update_params(False, **params)
        tol = sparams['tol']
        t1 = sparams['t1']
        t2 = sparams['t2']
        p = sparams['p']
        sigma = sparams['sigma']
        beta = sparams['beta']
        max_iter = int(sparams['max_iter'])
        armijo_max_iter = int(sparams['armijo_max_iter'])

        solver = sparams['solver']
        if solver is None:
            if sparams['sparse_hessian']:
                # solver = pypardiso.spsolve # pypardiso is MUCH faster then scipy.sparse.linalg.spsolve. E.g. pypardiso takes 5ms while scipy's spsolve takes 200ms.
                solver = scipy.sparse.linalg.spsolve
            else:
                solver = np.linalg.solve

        # To prevent possible changes to x_0
        xk = x_0.copy()

        func = self.func

        self.N = max_iter

        if documentation is None:
            documentation = self.documentation

        if documentation:
            doc = np.zeros((max_iter, 4))
            if save_xk:
                his_xk = np.zeros((max_iter, len(x_0)))

        alpha_k = 1.
        armijo_i = 1
        t_iter = tqdm.trange(max_iter, disable = not verbose, desc="Newton-Algorithm")
        for k in t_iter:
            func_xk = func(xk)
            grad_f = self.getDerivative(xk)
            norm_grad = np.linalg.norm(grad_f)
            t_iter.set_postfix({"dJ" : norm_grad,  "armijo_i" : armijo_i})

            if norm_grad <= tol:
                self.N = k+1
                if documentation:
                    normalized_res = norm_grad/grad_f.size
                    doc[k] = func_xk, norm_grad, np.asarray(alpha_k), normalized_res
                    if save_xk:
                        his_xk[k] = xk
                break

            hessian_f = self.getHessian(xk)

            sk = solver(hessian_f, -grad_f)
            sk = sk.reshape(grad_f.shape)
            s_norm = np.linalg.norm(sk)
            if -np.dot(grad_f.ravel(), sk.ravel()) >= min(t1, t2 * s_norm**p)*s_norm**2:
                dk = sk # Newton-step
            else:
                dk = -grad_f # Gradient-step

            alpha_k, armijo_i = armijo_line_search(func, grad_f, xk, dk,
                                                   beta=beta, sigma=sigma,
                                                   fxk=func_xk,
                                                   max_iter=armijo_max_iter)

            if documentation:
                normalized_res = norm_grad/grad_f.size
                # This is needed to be compatible with numpy and cupy
                # Offenbar ist diese Variante nicht mehr mit numpy verträglich
                # doc[k] = np.asarray([func_xk, norm_grad, np.asarray(alpha_k), normalized_res])

                doc[k] = func_xk , norm_grad, np.asarray(alpha_k), normalized_res
                if save_xk:
                    his_xk[k] = xk

            xk += alpha_k*dk

        if documentation:
            # self.history = pd.DataFrame(doc[:self.N], columns=["J(x)", "||J'(x)||", "alpha"])
            self.history = dict()
            self.history["J(x)"] = doc[:self.N,0]
            self.history["||J'(x)||"] = doc[:self.N,1]
            self.history["alpha"] = doc[:self.N,2]
            self.history["normalized residual"] = doc[:self.N,3]
            if save_xk:
                self.his_xk = his_xk[:self.N]

        self.total_N += k
        self.DJx = grad_f

        return xk

    def print_history(self, explicit = False):
        """Prints the history if called after the iteration.

        If explicit is True everything is printed. Otherwise pandas' print is used."""
        try:
            # I will never be able to make such a beautiful output like pandas
            import pandas as pd
            td = self.history
            df = pd.DataFrame(td)
            if explicit:
                print(df.to_string())
            else:
                print(df)
        except ModuleNotFoundError:
            a,b,c = self.history.keys()
            print(f"{'':<5} {a:<10} {b:<10} {c:<10}")
            for k in range(len(self.history['alpha'])):
                print(f"{k:<5} {self.history[a][k]:<10.5f} {self.history[b][k]:<10.5f} {self.history[c][k]:<10.2f}")

    __call__ = find_min


    @staticmethod
    def minimize(func : callable, x0 : np.ndarray, fprime : callable, hess_f : callable,*,
                 verbose : bool = False, **params):
        """
        The (global) newton algorithm with Armijo line search.

        Parameters
        ----------
        func : TYPE
            The function to minimize.
        fprime : function
            A method to provide the derivative of f at a point x0.
        hess_f : function
            A method to provide the hessian of f at a point x0.
        x_0 : ndarray
            starting point.
        verbose : boolean, optional
            determine if the loop should be shown in the console
        solver : function, optional
            A custom solver to solve the equation
            sk = solver(hessian_f, -grad_f)
            You set up everything to ensure that the function can be evaluated
            and gives the right result.
        params, optional
            additional parameters for the newton iteration.
            See "update_params" for more information.
        Returns
        -------
        xk : ndarray
            Result, which is the minimizing argument.

        """
        newt = Newton(func, fprime, hess_f, documentation = False, **params)
        return newt(x0, verbose = verbose, save_xk = False)

def minimize(func : callable, x0 : np.ndarray, fprime : callable, hess_f : callable,*,
             verbose : bool = False, **params):
    return Newton.minimize(func, x0, fprime, hess_f,
                 documentation = False, verbose = verbose,
                 save_xk = False, **params)

def draw_lines(x = None, y = None, ax = None, color = 'red'):
    assert len(x) == len(y)

    if ax is None:
        fig, ax = plt.subplots(1,1)

    # See https://stackoverflow.com/questions/35363444/plotting-lines-connecting-points, JinjerJohn
    xx = np.vstack([x[:-1], x[1:]])
    yy = np.vstack([y[:-1], y[1:]])

    ax.plot(xx, yy, marker='o', linestyle='dashed', color=color)
    return ax

if __name__ == '__main__':
    # An example
    function = 'rosenbrock'

    class ChungReynolds:
        def __call__(self, x):
            return np.square(x).sum()**2

        def getDerivative(self, x):
            return 4*np.square(x).sum() * x

        def getHessian(self, x):
            if len(x.shape) == 1:
                x = x.reshape((-1,1))
            hessian = 8*x*x.T
            diag = np.einsum('ii->i', hessian)
            diag += 4*np.square(x).sum()
            return hessian

    class Colville:
        def __call__(self, x):
            x_1, x_2, x_3, x_4 = x
            return (100*(x_1 - x_2**2)**2 + (1 - x_1)^2
                    + 90*(x_4 - x_3**2)**2 + (1 - x_3)**2
                    + 10.1*((x_2 - 1)^2 + (x_4 - 1)**2)
                    + 19.8*(x_2 - 1)*(x_4 - 1))

        def getDerivative(self, x):
            pass

    ## 2-D example.
    class Rosenbrock:
        def __init__(self, a=1.,b=100.):
            self.a = a
            self.b = b

        def __call__(self, x):
            x1 = x[0]
            x2 = x[1]

            return self.a*(1-x1)**2 + self.b*(x2 - x1**2)**2

        def getDerivative(self, x):
            x1 = x[0]
            x2 = x[1]
            return np.asarray([-2*(self.a*(1-x1) + 2*self.b*x1*(x2-x1**2)), 2*self.b*(x2-x1**2)], dtype=np.double)

        def getHessian(self, x):
            x1 = x[0]
            x2 = x[1]

            return np.asarray([[-2*(-self.a + 2*self.b*x2 - 6*self.b*x1**2), -4*self.b*x1],
                               [-4*self.b*x1, np.asarray(2*self.b)]])

        def plot(self):
            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d')

            xlist = np.linspace(-2,2, num=100)
            ylist = np.linspace(-0.6,3, num=100)

            X,Y = np.meshgrid(xlist, ylist)

            Z = self(np.asarray([X,Y]))
            # Z = np.sin(-(X+Y))*X

            ax.plot_surface(X,Y,Z, cmap='magma')
            # ax.plot(1,1,1, 'o', linewidth=10)s
            ax.plot([1.], [1.], [self(np.asarray([1.,1.]))], markerfacecolor='green', markeredgecolor='green', marker='o', markersize=5)
            # ax.plot_surface(1,1,self(np.asarray([1.,1.])))
            return fig, ax


    ## 1-D example.
    class Polynom:
        def __call__(self, x):
            return x**2 + 2*x


        def getDerivative(self, x):
            return 2*x + 2


        def getHessian(self, x):
            return np.array([[2.]])


        def plot(self):
            fig, ax = plt.subplots(1,1)

            xlist = np.linspace(-3,3, num=1009)
            ylist = self(xlist)
            # Z = np.sin(-(X+Y))*X

            ax.plot(xlist, ylist)
            # ax.plot([-0.35173371], [0.82718403], markerfacecolor='green', markeredgecolor='green',
            #         marker='o', markersize=5)
            return fig, ax


    class Pol:
        def __init__(self, a : float, n : int):
            assert n >= 1
            self.a = a
            self.n = int(n)


        def __call__(self, x):
            return 1/(self.n + 1) * (x + self.a)**(self.n + 1)


        def getDerivative(self, x):
            return (x + self.a)**self.n


        def getHessian(self, x):
            return np.array([self.n * (x + self.a)**(self.n - 1)], dtype=np.double)


        def plot(self):
            fig, ax = plt.subplots(1,1)

            xlist = np.linspace(-10,10, num=1009)
            ylist = self(xlist)
            # Z = np.sin(-(X+Y))*X

            ax.plot(xlist, ylist)
            # ax.plot([-0.35173371], [0.82718403], markerfacecolor='green', markeredgecolor='green',
            #         marker='o', markersize=5)
            return fig, ax


    if function.lower() == 'chungreynolds':
        start_x = np.asarray([1.2,1.1, 0.1]*60, dtype = np.double).reshape((-1,1))
        cr = ChungReynolds()
        newt = Newton(cr, cr.getDerivative, cr.getHessian, documentation=True)
        sol = newt.find_min(start_x, tol=1e-9)
        print(f"sol = {sol.ravel()}, f(sol) = {cr(sol)}, ||f'(sol)|| = {np.linalg.norm(cr.getDerivative(sol))}")
        newt.print_history()


    if function.lower() == 'rosenbrock':
        ex_x = np.asarray([-2,-0.5], dtype=np.double)
        ros = Rosenbrock()
        # ros.plot()
        newt = Newton(ros, ros.getDerivative, ros.getHessian, documentation=True)
        start_x = np.asarray([-1.2,1], dtype = np.double)
        newt.update_params(max_iter = 1e3, t1=1e-6, t2=1e-6, p=0.1, tol=1e-9, sigma=1e-4)
        sol = newt.find_min(start_x, tol = 1e-9)
        print(f"sol = {sol}, f(sol) = {ros(sol)}, ||f'(sol)|| = {np.linalg.norm(ros.getDerivative(sol))}")
        newt.print_history()


    if function.lower() == 'polynomial':
        import seaborn as sns
        sns.set_theme()
        start_x = np.asarray([-3.])
        pol = Pol(2, 3)
        newt = Newton(pol, pol.getDerivative, pol.getHessian, documentation=True,
                      )
        sol = newt.find_min(start_x, max_iter=100, tol=1e-13, save_xk = True)
        print(f"sol = {sol}, f(sol) = {pol(sol)}, ||f'(sol)|| = {np.linalg.norm(pol.getDerivative(sol))}")
        newt.print_history()
        fig, ax = pol.plot()
        draw_lines(newt.his_xk.ravel(), newt.history['J(x)'], ax, 'r')










