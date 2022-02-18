
# Dormann & Prince 853 implementation

# Chirag Falor
# February 2022

# This code heavily borrows from the Dopri5 implementation by Jolyon Bloomfield

import numpy as np
import math

class DopriIntegrationError(Exception):
    """Error class for integration"""
    pass

class DOPRI853(object):
    """Dormand-Prince 5th order integrator"""

    def __init__(self,
                 t0,                # Starting time
                 init_values,       # Starting values
                 derivs,            # Derivative function
                 init_h=0.01,       # Initial step size
                 min_h=5e-8,        # Minimum step size
                 max_h=1.0,         # Maximum step size
                 rtol=1e-7,         # Relative tolerance
                 atol=1e-7,         # Absolute tolerance
                 params=None):      # Parameters to pass to the derivatives function
        """
        Initialize the integrator
        """
        self.derivs = derivs
        self.values = init_values
        self.t = t0
        self.hnext = init_h  # Step we're about to take
        self.max_h = max_h
        self.min_h = min_h
        self.rtol = rtol
        self.atol = atol
        self.params = params

        # Internal variables
        self.hdid = 0        # Previous step we just took
        self.dxdt = None    # Used for FSAL
        
    def set_init_values(self, t0, init_values):
        self.values = init_values
        self.t = t0

    def update_max_h(self, new_max_h):
        """Updates the max step size"""
        if new_max_h < self.min_h:
            raise DopriIntegrationError("Requested max step size less than min step size")
        self.max_h = new_max_h
        self.hnext = min(self.hnext, self.max_h)

    def clear_fsal(self):
        """Clears FSAL information, forcing it to be recalculated"""
        self.dxdt = None

    def step(self, newtime):
        """Take a step"""
        rejected = False

        while True:
            # Comment these two lines out if you want to allow it to go past newtime
            if self.t + self.hnext > newtime:
                self.hnext = newtime - self.t
            self._take_step(self.hnext)
            if self._good_step(rejected):
                break
            else:
                rejected = True

        # Update our data
        self.t += self.hdid
        self.values = self.newvalues
        self.dxdt = self.newdxdt

    def _good_step(self, rejected, minscale=0.2, maxscale=5, safety=0.8):
        """
        Checks if the previous step was good, and updates the step size
        rejected stores whether or not we rejected a previous attempt at this step
        minscale is the minimum we will scale down the stepsize
        maxscale is the maximum we will scale up the stepsize
        safety is the safety factor in the stepsize estimation
        """
        alpha = 1/8
        # Compute the scaled error of the past step
        err = self._error()
        if err <= 1.0:
            # Step was good
            # Figure out how to scale our next step
            if err == 0.0:
                scale = maxscale
            else:
                scale = safety * math.pow(err, -alpha)
                scale = max(scale, minscale)
                scale = min(scale, maxscale)
            # Make sure we're not increasing the step if we just rejected something
            if rejected:
                scale = min(scale, 1.0)
            # Update the step sizes
            self.hdid = self.hnext
            self.hnext = self.hdid * scale
            self.hnext = min(self.hnext, self.max_h)
            self.hnext = max(self.hnext, self.min_h)
            return True
        else:
            # Error was too big
            if self.hnext == self.min_h:
                raise DopriIntegrationError("Step size decreased below minimum threshold")
            # Try again!
            scale = max(safety * math.pow(err, -alpha), minscale)
            self.hnext *= scale
            self.hnext = max(self.hnext, self.min_h)
            return False

    def _error(self):
        """Computes the normalized error in the step just taken"""
        maxed = np.column_stack((np.abs(self.values), np.abs(self.newvalues)))
        maxed = np.max(maxed, axis=1)
        sk = self.atol + self.rtol * maxed
        norm_err_1 = (np.linalg.norm(self.errors1 / sk))**2
        norm_err_2 = (np.linalg.norm(self.errors2 / sk))**2
        deno = norm_err_1 + 0.01*norm_err_2
        if deno <= 0:
            deno = 1
        return norm_err_1 * math.sqrt(1/(len(self.errors1)*deno))

    # Coefficients for DOPRI853
    # We use the following coefficients for implementation of Dormand-Prince 853
    c2  = 0.526001519587677318785587544488e-01
    c3  = 0.789002279381515978178381316732e-01
    c4  = 0.118350341907227396726757197510e+00
    c5  = 0.281649658092772603273242802490e+00
    c6  = 0.333333333333333333333333333333e+00
    c7  = 0.25e+00
    c8  = 0.307692307692307692307692307692e+00
    c9  = 0.651282051282051282051282051282e+00
    c10 = 0.6e+00
    c11 = 0.857142857142857142857142857142e+00
    c14 = 0.1e+00
    c15 = 0.2e+00
    c16 = 0.777777777777777777777777777778e+00

    b1 =   5.42937341165687622380535766363e-2
    b6 =   4.45031289275240888144113950566e0
    b7 =   1.89151789931450038304281599044e0
    b8 =  -5.8012039600105847814672114227e0
    b9 =   3.1116436695781989440891606237e-1
    b10 = -1.52160949662516078556178806805e-1
    b11 =  2.01365400804030348374776537501e-1
    b12 =  4.47106157277725905176885569043e-2

    bhh1 = 0.244094488188976377952755905512e+00
    bhh2 = 0.733846688281611857341361741547e+00
    bhh3 = 0.220588235294117647058823529412e-01

    er1  =  0.1312004499419488073250102996e-01
    er6  = -0.1225156446376204440720569753e+01
    er7  = -0.4957589496572501915214079952e+00
    er8  =  0.1664377182454986536961530415e+01
    er9  = -0.3503288487499736816886487290e+00
    er10 =  0.3341791187130174790297318841e+00
    er11 =  0.8192320648511571246570742613e-01
    er12 = -0.2235530786388629525884427845e-01

    a21 =    5.26001519587677318785587544488e-2
    a31 =    1.97250569845378994544595329183e-2
    a32 =    5.91751709536136983633785987549e-2
    a41 =    2.95875854768068491816892993775e-2
    a43 =    8.87627564304205475450678981324e-2
    a51 =    2.41365134159266685502369798665e-1
    a53 =   -8.84549479328286085344864962717e-1
    a54 =    9.24834003261792003115737966543e-1
    a61 =    3.7037037037037037037037037037e-2
    a64 =    1.70828608729473871279604482173e-1
    a65 =    1.25467687566822425016691814123e-1
    a71 =    3.7109375e-2
    a74 =    1.70252211019544039314978060272e-1
    a75 =    6.02165389804559606850219397283e-2
    a76 =   -1.7578125e-2

    a81 =    3.70920001185047927108779319836e-2
    a84 =    1.70383925712239993810214054705e-1
    a85 =    1.07262030446373284651809199168e-1
    a86 =   -1.53194377486244017527936158236e-2
    a87 =    8.27378916381402288758473766002e-3
    a91 =    6.24110958716075717114429577812e-1
    a94 =   -3.36089262944694129406857109825e0
    a95 =   -8.68219346841726006818189891453e-1
    a96 =    2.75920996994467083049415600797e1
    a97 =    2.01540675504778934086186788979e1
    a98 =   -4.34898841810699588477366255144e1
    a101 =   4.77662536438264365890433908527e-1
    a104 =  -2.48811461997166764192642586468e0
    a105 =  -5.90290826836842996371446475743e-1
    a106 =   2.12300514481811942347288949897e1
    a107 =   1.52792336328824235832596922938e1
    a108 =  -3.32882109689848629194453265587e1
    a109 =  -2.03312017085086261358222928593e-2

    a111 =  -9.3714243008598732571704021658e-1
    a114 =   5.18637242884406370830023853209e0
    a115 =   1.09143734899672957818500254654e0
    a116 =  -8.14978701074692612513997267357e0
    a117 =  -1.85200656599969598641566180701e1
    a118 =   2.27394870993505042818970056734e1
    a119 =   2.49360555267965238987089396762e0
    a1110 = -3.0467644718982195003823669022e0
    a121 =   2.27331014751653820792359768449e0
    a124 =  -1.05344954667372501984066689879e1
    a125 =  -2.00087205822486249909675718444e0
    a126 =  -1.79589318631187989172765950534e1
    a127 =   2.79488845294199600508499808837e1
    a128 =  -2.85899827713502369474065508674e0
    a129 =  -8.87285693353062954433549289258e0
    a1210 =  1.23605671757943030647266201528e1
    a1211 =  6.43392746015763530355970484046e-1

    a141 =  5.61675022830479523392909219681e-2
    a147 =  2.53500210216624811088794765333e-1
    a148 = -2.46239037470802489917441475441e-1
    a149 = -1.24191423263816360469010140626e-1
    a1410 =  1.5329179827876569731206322685e-1
    a1411 =  8.20105229563468988491666602057e-3
    a1412 =  7.56789766054569976138603589584e-3
    a1413 = -8.298e-3

    a151 =  3.18346481635021405060768473261e-2
    a156 =  2.83009096723667755288322961402e-2
    a157 =  5.35419883074385676223797384372e-2
    a158 = -5.49237485713909884646569340306e-2
    a1511 = -1.08347328697249322858509316994e-4
    a1512 =  3.82571090835658412954920192323e-4
    a1513 = -3.40465008687404560802977114492e-4
    a1514 =  1.41312443674632500278074618366e-1
    a161 = -4.28896301583791923408573538692e-1
    a166 = -4.69762141536116384314449447206e0
    a167 =  7.68342119606259904184240953878e0
    a168 =  4.06898981839711007970213554331e0
    a169 =  3.56727187455281109270669543021e-1
    a1613 = -1.39902416515901462129418009734e-3
    a1614 =  2.9475147891527723389556272149e0
    a1615 = -9.15095847217987001081870187138e0

    d41  = -0.84289382761090128651353491142e+01
    d46  =  0.56671495351937776962531783590e+00
    d47  = -0.30689499459498916912797304727e+01
    d48  =  0.23846676565120698287728149680e+01
    d49  =  0.21170345824450282767155149946e+01
    d410 = -0.87139158377797299206789907490e+00
    d411 =  0.22404374302607882758541771650e+01
    d412 =  0.63157877876946881815570249290e+00
    d413 = -0.88990336451333310820698117400e-01
    d414 =  0.18148505520854727256656404962e+02
    d415 = -0.91946323924783554000451984436e+01
    d416 = -0.44360363875948939664310572000e+01

    d51  =  0.10427508642579134603413151009e+02
    d56  =  0.24228349177525818288430175319e+03
    d57  =  0.16520045171727028198505394887e+03
    d58  = -0.37454675472269020279518312152e+03
    d78  =  0.35763911791061412378285349910e+03
    d79  =  0.93405324183624310003907691704e+02
    d710 = -0.37458323136451633156875139351e+02
    d711 =  0.10409964950896230045147246184e+03
    d712 =  0.29840293426660503123344363579e+02
    d713 = -0.43533456590011143754432175058e+02
    d714 =  0.96324553959188282948394950600e+02
    d715 = -0.39177261675615439165231486172e+02
    d716 = -0.14972683625798562581422125276e+03

    def _take_step(self, h):
        '''
        Takes an individual step of size h according to Dopr853
        '''
        global c2,c3,c4,c5,c6,c7,c8,c9,c10,c11,c14,c15,c16, b1,b6,b7,b8,b9,b10,b11,b12,bhh1,bhh2,bhh3, er1,er6,er7,er8,er9,er10,er11,er12
        global a21,a31,a32,a41,a43,a51,a53,a54,a61,a64,a65,a71,a74,a75,a76,a81,a84,a85,a86,a87,a91,a94,a95,a96,a97,a98,a101,a104,a105,a106,a107,a108,a109,a111,a114,a115,a116,a117,a118,a119,a1110, a121,a124,a125,a126,a127,a128,a129,a1210,a1211,a141,a147,a148, a149,a1410,a1411,a1412,a1413,a151,a156,a157,a158,a1511,a1512,a1513,a1514,a161,a166,a167,a168,a169,a1613,a1614,a1615
        global d41,d46,d47,d48,d49,d410,d411,d412,d413,d414,d415,d416,d51,d56,d57,d58,d59,d510,d511,d512,d513,d514,d515,d516,d61,d66,d67,d68,d69,d610,d611,d612,d613,d614,d615,d616,d71,d76,d77,d78,d79,d710,d711,d712,d713,d714,d715,d716
        # Initialize
        if self.dxdt is None:
            self.dxdt = self.derivs(self.t, self.values, self.params)

        # Initialize the Runge-Kutta residual storage
        k = [None] * 12
        k[0] = self.dxdt
        # Compute the first intermediate step

        newvals = self.values + h*self.a21*k[0]
        k[1] = self.derivs(self.t+self.c2*h, newvals, self.params)

        newvals = self.values + h*(self.a31*k[0]+self.a32*k[1])
        k[2] = self.derivs(self.t+self.c3*h, newvals, self.params)

        newvals = self.values + h*(self.a41*k[0]+self.a43*k[2])
        k[3] = self.derivs(self.t+self.c4*h, newvals, self.params)

        newvals = self.values + h*(self.a51*k[0]+self.a53*k[2]+self.a54*k[3])
        k[4] = self.derivs(self.t+self.c5*h, newvals, self.params)

        newvals = self.values + h*(self.a61*k[0]+self.a64*k[3]+self.a65*k[4])
        k[5] = self.derivs(self.t+self.c6*h, newvals, self.params)

        newvals = self.values + h*(self.a71*k[0]+self.a74*k[3]+self.a75*k[4]+self.a76*k[5])
        k[6] = self.derivs(self.t+self.c7*h, newvals, self.params)

        newvals = self.values + h*(self.a81*k[0]+self.a84*k[3]+self.a85*k[4]+self.a86*k[5]+self.a87*k[6])
        k[7] = self.derivs(self.t+self.c8*h, newvals, self.params)

        newvals = self.values + h*(self.a91*k[0]+self.a94*k[3]+self.a95*k[4]+self.a96*k[5]+self.a97*k[6]+self.a98*k[7])
        k[8] = self.derivs(self.t+self.c9*h, newvals, self.params)

        newvals = self.values + h*(self.a101*k[0]+self.a104*k[3]+self.a105*k[4]+self.a106*k[5]+self.a107*k[6]+self.a108*k[7]+self.a109*k[8])
        k[9] = self.derivs(self.t+self.c10*h, newvals, self.params)

        newvals = self.values + h*(self.a111*k[0]+self.a114*k[3]+self.a115*k[4]+self.a116*k[5]+self.a117*k[6]+self.a118*k[7]+self.a119*k[8]+self.a1110*k[9])
        k[10] = self.derivs(self.t+self.c11*h, newvals, self.params)
        # k2 is k[10] as per the Numerical Recipes code
        # k3 is k[11] as per the Numerical Recipes code
        newvals = self.values + h*(self.a121*k[0]+self.a124*k[3]+self.a125*k[4]+self.a126*k[5]+self.a127*k[6]+self.a128*k[7]+self.a129*k[8]+self.a1210*k[9]+self.a1211*k[10])
        k[11] = self.derivs(self.t+h, newvals, self.params)

        final_slope = self.b1*k[0]+self.b6*k[5]+self.b7*k[6]+self.b8*k[7]+self.b9*k[8]+self.b10*k[9]+self.b11*k[10]+self.b12*k[11]

        # save the results
        self.newvalues = self.values + h*(final_slope)
        self.newdxdt = self.derivs(self.t+h, self.newvalues, self.params)

        # Compute the errors
        self.errors1 = h * (final_slope - self.bhh1*k[0] - self.bhh2*k[8] - self.bhh3 * k[11])
        self.errors2 = h * (self.er1*k[0] + self.er6*k[5] + self.er7*k[6] + self.er8*k[7] + self.er9*k[8] + self.er10*k[9] + self.er11*k[10] + self.er12*k[11])
        

    def _take_step_dopr5(self, h):
        """Take an individual step with size h"""
        # Check that we're initialized
        if self.dxdt is None:
            self.dxdt = self.derivs(self.t, self.values, self.params)

        # Compute the slopes and updated positions
        slopes = [None] * 7
        slopes[0] = self.dxdt   # stored from previous step
        newvals = self.values + h*self._dopri5coeffs[1][0]*slopes[0]
        slopes[1] = self.derivs(self.t + h*self._dopri5times[1], newvals, self.params)
        for i in range(2, 7):
            newvals = self.values + h*sum(self._dopri5coeffs[i][j]*slopes[j] for j in range(i))
            slopes[i] = self.derivs(self.t + h*self._dopri5times[i], newvals, self.params)

        # Save the results
        self.newvalues = newvals
        self.newdxdt = slopes[6]
        # Compute the errors
        self.errors = h * sum(self._dopri5errors[i]*slopes[i] for i in range(7))







