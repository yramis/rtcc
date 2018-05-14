import numpy as np

class helper_diis(object):
    def __init__(self,x1,x2,max_diis):

        self.x1_old     = x1.copy()
        self.x2_old     = x2.copy()
        self.max_diis   = max_diis

        # Setup DIIS
        self.diis_values_x1 = [x1.copy()]
        self.diis_values_x2 = [x2.copy()]
        self.diis_errors    = []
        self.diis_size      = 0

    def add_error_vector(self,x1,x2):

        # Add DIIS vector
        self.diis_values_x1.append(x1.copy())
        self.diis_values_x2.append(x2.copy())

        # Add new error vector
        error_x1 = (self.diis_values_x1[-1] - self.x1_old).ravel()
        error_x2 = (self.diis_values_x2[-1] - self.x2_old).ravel()
        self.diis_errors.append(np.concatenate((error_x1,error_x2)))
        self.x1_old = x1.copy()
        self.x2_old = x2.copy()

    def extrapolate(self,x1,x2):

        # Limit size of DIIS vector
        if (len(self.diis_values_x1) > self.max_diis+1):
            del self.diis_values_x1[0]
            del self.diis_values_x2[0]
            del self.diis_errors[0]

        self.diis_size = len(self.diis_values_x1) - 1

        # Build error matrix B
        B = np.ones((self.diis_size+1,self.diis_size+1))*-1
        B[-1,-1] = 0

        for n1,e1 in enumerate(self.diis_errors):
            B[n1,n1] = np.dot(e1,e1)
            for n2,e2 in enumerate(self.diis_errors):
                if n1 >= n2: continue
                B[n1,n2] = np.dot(e1,e2)
                B[n2,n1] = B[n1,n2]

        B[:-1,:-1] /= np.abs(B[:-1,:-1]).max()

        # Build residual vector
        resid = np.zeros(self.diis_size+1)
        resid[-1] = -1

        # Solve pulay equations
        ci = np.linalg.solve(B,resid)

        # Calculate new amplitudes
        x1 = np.zeros_like(self.x1_old)
        x2 = np.zeros_like(self.x2_old)
        for num in range(self.diis_size):
            x1 += ci[num] * self.diis_values_x1[num + 1]
            x2 += ci[num] * self.diis_values_x2[num + 1]

        # Save extrapolated amplitudes to old_t amplitudes
        self.x1_old = x1.copy()
        self.x2_old = x2.copy()

        return x1,x2
