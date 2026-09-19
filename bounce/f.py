import numpy as np

class F_ainv:
    def fitter( self, x, c0, c1_0, c1_1, c2_01, c2_11 ):
        beta = x[0]
        mqhat = x[1]
        return c0 + c1_0*beta + c1_1*mqhat + c2_01*beta*mqhat + c2_11*mqhat*mqhat

    def __init__( self, epsilon=1.0 ):
        self.fp_ainv = np.loadtxt( "../global_fit/coeffs_ainv_beta_mqhat.dat" )
        self.fcp_ainv = np.loadtxt( "../global_fit/cov_ainv_beta_mqhat.dat" )
        self.Df_ainvs = [self.Df0_ainv, self.Df1_ainv, self.Df2_ainv, self.Df3_ainv, self.Df4_ainv]
        self.epsilon = epsilon

    def __call__( self, beta_, mq_ ):
        return self.fitter( [beta_, mq_], self.fp_ainv[0], self.fp_ainv[1], self.fp_ainv[2], self.fp_ainv[3], self.fp_ainv[4] )

    # check
    # f_ainv( 10.95, 0.19 )
    
    def Df0_ainv( self, beta_, mq_ ):
        return 1.0
    
    def Df1_ainv( self, beta_, mq_ ):
        return beta_
    
    def Df2_ainv( self, beta_, mq_ ):
        return mq_
    
    def Df3_ainv( self, beta_, mq_ ):
        return beta_*mq_
    
    def Df4_ainv( self, beta_, mq_ ):
        return mq_**2

    def D( self, beta_, mq_ ):
        df = np.array([ f(beta_,mq_) for f in self.Df_ainvs ])
        return np.sqrt( df@self.fcp_ainv@df )*self.epsilon

class F_TmTc:
    def fitter( self, x, c1_0, c2_01 ):
        dbeta = x[0]
        mq = x[1]
        return c1_0*dbeta + c2_01*dbeta*mq

    # fp_ainv = np.loadtxt( "../global_fit/coeffs_ainv_beta_mqhat.dat" )
    # fcp_ainv = np.loadtxt( "../global_fit/cov_ainv_beta_mqhat.dat" )

    def __init__( self, Nt, epsilon=1.0 ):
        self.fp_ainv = np.loadtxt( "../global_fit/coeffs_ainv_beta_mqhat.dat" )
        self.fcp_ainv = np.loadtxt( "../global_fit/cov_ainv_beta_mqhat.dat" )
        self.Df_ainvs = [self.Df1_ainv, self.Df3_ainv]
        self.Nt = Nt
        self.epsilon = epsilon

    def __call__( self, beta_, mq_ ):
        return self.fitter( [beta_, mq_], self.fp_ainv[1], self.fp_ainv[3] )/self.Nt

    # check
    # f_ainv( 10.95, 0.19 )
    
    def Df1_ainv( self, dbeta, mq ):
        return dbeta
        
    def Df3_ainv( self, dbeta, mq ):
        return dbeta*mq
    

    def D( self, dbeta, mq ):
        df = np.array([ f(dbeta, mq) for f in self.Df_ainvs ])
        # print( df.shape )
        cov = self.fcp_ainv[ [1,3] ].T[ [1,3] ]
        # print( cov.shape )
        return np.sqrt( df@ cov @df )/self.Nt*self.epsilon

# check
# ii=3

# eps=1.0e-7
# beta_=10.95
# mq_=0.19

# a = Df_ainvs[ii]( beta_, mq_ )
# fp_ainvP = np.copy(fp_ainv)
# fp_ainvM = np.copy(fp_ainv)
# fp_ainvP[ii] += eps
# fp_ainvM[ii] -= eps

# b = ( fitter( np.array([beta_, mq_]), fp_ainvP[0], fp_ainvP[1], fp_ainvP[2], fp_ainvP[3], fp_ainvP[4], fp_ainvP[5] )-fitter( np.array([beta_, mq_]), fp_ainvM[0], fp_ainvM[1], fp_ainvM[2], fp_ainvM[3], fp_ainvM[4], fp_ainvM[5] ))/(2.0*eps)
# a, b



#####################################################


class F_MBsub:
    def fitter( self, x, c0, c1, c2, c3):
        asq = x[0]
        M = x[1]
        return c0 + c1*asq + c2*M + c3*M*M

    def __init__( self, epsilon=1.0 ):
        self.fp_MB = np.loadtxt( "../global_fit/coeffs_MB_asq_mq.dat" )
        self.fcp_MB = np.loadtxt( "../global_fit/cov_MB_asq_mq.dat" )
        self.Df_MBs = [self.Df0, self.Df1, self.Df2, self.Df3]
        self.epsilon = epsilon

    def __call__( self, asq_, mq_ ):
        return self.fitter( [asq_, mq_], self.fp_MB[0], self.fp_MB[1], self.fp_MB[2], self.fp_MB[3])
    
    def Df0( self, asq_, mq_ ):
        return 1.0
    
    def Df1( self, asq_, mq_ ):
        return asq_
    
    def Df2( self, asq_, mq_ ):
        return mq_

    def Df3( self, asq_, mq_ ):
        return mq_**2

    def D( self, asq_, mq_ ):
        df = np.array([ f(asq_,mq_) for f in self.Df_MBs ])
        # print( df )
        # print( self.fcp_MB )
        return np.sqrt( df@self.fcp_MB@df )*self.epsilon

    def D1(self, asq_, mq_):
        return self.fp_MB[1]

    def D2(self, asq_, mq_):
        return self.fp_MB[2] + 2.0*mq_*self.fp_MB[3]



# #####################################################


# class F_MBsub:
#     def fitter( self, x, c0, c1, c11, c2 ):
#         asq = x[0]
#         M = x[1]
#         return c0 + c1*asq + c11*asq**2 + c2*M

#     def __init__( self, epsilon=1.0 ):
#         self.fp_MB = np.loadtxt( "../global_fit/coeffs_MB_asq_mq.dat" )
#         self.fcp_MB = np.loadtxt( "../global_fit/cov_MB_asq_mq.dat" )
#         self.Df_MBs = [self.Df0, self.Df1, self.Df2, self.Df3]
#         self.epsilon = epsilon

#     def __call__( self, asq_, mq_ ):
#         return self.fitter( [asq_, mq_], self.fp_MB[0], self.fp_MB[1], self.fp_MB[2], self.fp_MB[3])
    
#     def Df0( self, asq_, mq_ ):
#         return 1.0
    
#     def Df1( self, asq_, mq_ ):
#         return asq_
    
#     def Df2( self, asq_, mq_ ):
#         return asq_**2

#     def Df3( self, asq_, mq_ ):
#         return mq_

#     def D( self, asq_, mq_ ):
#         df = np.array([ f(asq_,mq_) for f in self.Df_MBs ])
#         return np.sqrt( df@self.fcp_MB@df )*self.epsilon

#     def D1(self, asq_, mq_):
#         return self.fp_MB[1] + 2.0*self.fp_MB[2]*asq_

#     def D2(self, asq_, mq_):
#         return self.fp_MB[3]


#####################################################


class F_MB:
    def __init__( self, epsilon=1.0 ):
        self.f_ainv = F_ainv(epsilon)
        self.f_MBsub = F_MBsub(epsilon)

    def __call__( self, beta_, mqhat_ ):
        ainv = self.f_ainv( beta_, mqhat_ )
        asq = 1.0/ainv**2
        mq = mqhat_ * ainv
        return self.f_MBsub( asq, mq )

    def D( self, beta_, mqhat_ ):
        ainv = self.f_ainv( beta_, mqhat_ )
        dainv = self.f_ainv.D( beta_, mqhat_ )

        asq = 1.0/ainv**2
        mq = mqhat_ * ainv

        dMB_dainv = -2.0/ainv**3 * self.f_MBsub.D1( asq, mq ) + mqhat_ * self.f_MBsub.D2( asq, mq )        
        df1 = np.abs(dMB_dainv)*dainv

        df2 = self.f_MBsub.D( asq, mq )
        
        return np.sqrt( df1**2 + df2**2 )



# # check
# ii=2

# eps=1.0e-7
# beta_=10.95
# mq_=0.19

# a = Df_MBs[ii]( beta_, mq_ )
# fp_MBP = np.copy(fp_MB)
# fp_MBM = np.copy(fp_MB)
# fp_MBP[ii] += eps
# fp_MBM[ii] -= eps

# b = ( fitter( np.array([beta_, mq_]), fp_MBP[0], fp_MBP[1], fp_MBP[2], fp_MBP[3], fp_MBP[4], fp_MBP[5] )-fitter( np.array([beta_, mq_]), fp_MBM[0], fp_MBM[1], fp_MBM[2], fp_MBM[3], fp_MBM[4], fp_MBM[5] ))/(2.0*eps)
# a, b


# def Df_MB( beta_, mq_ ):
#     df = np.array([ f(beta_,mq_) for f in Df_MBs ])
#     return np.sqrt( df@fcp_MB@df )