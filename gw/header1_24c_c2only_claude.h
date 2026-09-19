#pragma once



const int nparallel=4;



std::map<std::string, int> nbetas = {
  { "0p2000", 9},
  { "0p3000", 8},
  { "0p4000", 8}
};



const int nbins=40;




struct F_Tc {
  V fp;

  F_Tc(const V fp_)
    : fp(fp_)
  {}

  double operator() (const double MB) const {
    return fitter( MB, fp[0], fp[1] );
  }

  double fitter( const double MB, const double b0, const double b1 ) const {
    return b0+b1*MB;
  }
};



struct F_DeltaV{
  V fp;

  F_DeltaV(const V fp_)
    : fp(fp_)
  {}

  double operator() (const double TmTc, const double MB) const {
    return fitter( TmTc, MB, fp[0], fp[1], fp[2], fp[3] );
  }

  double fitter( const double TmTc, const double MB,
                 const double a0,
                 const double b0,
                 const double c0,
                 const double d0
                 ) const {
    return TmTc * ( a0 + b0*MB ) + TmTc*TmTc * ( c0 + d0*MB );
  }

  double dT( const double TmTc, const double MB ) const {
    const double a0 = fp[0];
    const double b0 = fp[1];
    const double c0 = fp[2];
    const double d0 = fp[3];
    return ( a0 + b0*MB ) + 2.0*TmTc * ( c0 + d0*MB );
  }
};


struct F_S3hat{
  V fp;   // stored [Mc,c0,c1,c2,gam]; c2-only ansatz uses ONLY Mc=fp[0], c2=fp[3], gam=fp[4]

  F_S3hat(const V fp_)
    : fp(fp_)
  {}

  double operator() (const double TmTc, const double MB) const {
    return fitter( TmTc, MB, fp[0], fp[3], fp[4] );
  }

  // c2-only thin-wall:  S3 = (MB-Mc)^gam * c2 / (T-Tc)^2      (c0 = c1 = 0)
  double fitter( const double TmTc, const double MB,
                 const double Mc,
                 const double c2,
                 const double gam ) const {
    assert(MB>Mc);
    return std::pow(MB-Mc, gam) * ( c2/TmTc/TmTc );
  }

  // dS3/d(TmTc) = (MB-Mc)^gam * ( -2 c2 / (T-Tc)^3 )
  double dT( const double TmTc, const double MB ) const {
    const double Mc = fp[0];
    const double c2 = fp[3];
    const double gam = fp[4];
    assert(MB>Mc);
    return std::pow(MB-Mc, gam) * ( -2.0 * c2/std::pow(TmTc,3) );
  }
};


// using Func=std::function<double(const double, const double)>;

// template<class F_S3>
struct F_Gammahat {
  const F_S3hat& f_S3;

  F_Gammahat(const F_S3hat& f_S3_)
    : f_S3(f_S3_)
  {}

  double operator()( const double TmTc, const double MB ) const {
    // if(std::abs(TmTc)<1.0e-13) return 0.0;
    const double S3 = f_S3(TmTc, MB);
    // std::cout << "TmTc,MB = " << TmTc << " " << MB << std::endl;
    // std::cout << "S3 = " << S3 << std::endl;
    assert( S3 > 0 );
    if(S3>1.0e20) return 0.0;
    const double coeff = std::pow( S3/(2.0*M_PI), 1.5 );
    const double efactor = std::exp(-S3);
    return coeff * efactor;
  }

  double dT( const double TmTc, const double MB ) const {
    const double S3 =  f_S3(TmTc, MB);
    if(S3>1.0e20) return 0.0;
    const double DS3 = f_S3.dT(TmTc, MB);

    const double coeff = std::pow( S3/(2.0*M_PI), 1.5 );
    const double dcoeff = 1.5*std::pow( S3/(2.0*M_PI), 0.5 )/(2.0*M_PI);
    const double efactor = std::exp(-S3);
    const double defactor = -std::exp(-S3);

    return (dcoeff*efactor + coeff*defactor)*DS3;
  }
};



double g_v_wall = -1.0;   // <0 = physical CJ vJ; >0 = fixed wall velocity from CLI (argv[2])

double getv(const double alpha){
  assert(alpha>0.0);
  if( g_v_wall > 0.0 ) return g_v_wall;   // fixed wall velocity v from CLI
  const double numer = std::sqrt(2.0*alpha/3.0 + alpha*alpha) + std::sqrt(1.0/3.0);
  const double denom = 1.0+alpha;
  return numer/denom;
}

// double getv(const double alpha){
//   return 1.0;
// }

// double getv(const double alpha){
//   return 0.2;
// }

// double get_alpha(const double T,
//                  const double Tc,
//                  const double MB,
//                  const F_DeltaV& f_DV,
//                  const F_S3hat& f_S3 ){
//   const double DVPrime = f_DV( T-Tc, MB );
//   const double dDVPrime_dT = f_DV.dT( T-Tc, MB );

//   const double numer = T*dDVPrime_dT;
//   const double denom = rhoRprime(T);

//   return (1.0/3.0) * numer/denom;
// }

double get_alpha(const double T,
                 const double Tc,
                 const double MB,
                 const F_DeltaV& f_DV,
                 const F_S3hat& f_S3 ){
  const double DVPrime = f_DV( T-Tc, MB );
  const double dDVPrime_dT = f_DV.dT( T-Tc, MB );

  const double numer = T*dDVPrime_dT;
  const double denom = T*dDVPrime_dT + 4.0*DVPrime;

  // std::cout << "debug9. " << T-Tc << " " << MB << " " << DVPrime << " " << dDVPrime_dT << std::endl;
  // std::cout << "debug10. " << numer << " " << denom << std::endl;
  return (1.0/3.0) * numer/denom;
}





void get_gw_params( double& alpha,
                    double& vJ,
                    double& beta_tilde,
                    double& Tstar,
                    double& Tc,
                    const F_Tc& f_Tc,
                    const F_DeltaV& f_DV,
                    const F_S3hat& f_S3,
                    const double MB,
                    const double MP,
                    const double sqrt_t0_inv_GeV,
                    const int NTs = 100, // -> 400
                    const double DTc = 0.0004 // # from DVPrime
                    ){
  Tc = f_Tc(MB);

  std::vector<double> Ts(NTs+1);
  std::vector<double> TmTcs(NTs+1);
  const double dT = DTc/NTs;
  for(int i=0; i<=NTs; i++) {
    Ts[i] = Tc-DTc + i*dT;
    TmTcs[i] = -DTc + i*dT;
  }

  const F_Gammahat f_Gammahat = F_Gammahat( f_S3 );

  std::vector<double> GamPrimes(NTs+1), DGamPrimes(NTs+1);
  std::vector<double> alphas(NTs+1), vJs(NTs+1);
  std::vector<double> Hsqs(NTs+1), Hinvs(NTs+1);

  for(int i=0; i<=NTs; i++) {
    GamPrimes[i] = f_Gammahat( TmTcs[i], MB );
    DGamPrimes[i] = f_Gammahat.dT( TmTcs[i], MB );
    assert( std::isfinite(GamPrimes[i]) );
    assert( std::isfinite(DGamPrimes[i]) );

    Hsqs[i] = std::pow(Ts[i],4)/(3.0*MP*MP) * ( M_PI*M_PI/30.0 * gstar(sqrt_t0_inv_GeV*Ts[i]) );
    Hinvs[i] = 1.0/std::sqrt(Hsqs[i]);

    alphas[i] = get_alpha(Ts[i], Tc, MB, f_DV, f_S3 );
    vJs[i] = getv(alphas[i]);
  }

  std::vector<std::vector<double>> T_T1_Ints(NTs+1); // [iT][iDT1] // 0 <= DT1 < Tc-T
  for(int iT=0; iT<=NTs; iT++) { // 0<=T<Tc
    std::vector<double> Ints(NTs-iT); // = 0.0;
    for(int iDT1=0; iDT1<NTs-iT; iDT1++){ // 0 <= DT1 < Tc-T
      double Int = 0.0;
      for(int iT2=iT; iT2<iT+iDT1; iT2++) {
        Int += 0.5*dT * Hinvs[iT2] * vJs[iT2];
        Int += 0.5*dT * Hinvs[iT2+1] * vJs[iT2+1];
      }
      Ints[iDT1]=Int;
    }
    T_T1_Ints[iT]=Ints;
  }

  std::vector<double> Is(NTs);


  for(int iT=0; iT<NTs; iT++) {
    double integral = 0.0;
    for(int iDT1=0; iDT1<NTs-iT; iDT1++) {
      const int iT1prime = iT+iDT1;
      integral += 0.5*dT * Hinvs[iT1prime] * GamPrimes[iT1prime] * std::pow(T_T1_Ints[iT][iDT1], 3);
      integral += 0.5*dT * Hinvs[iT1prime+1] * GamPrimes[iT1prime+1] * std::pow(T_T1_Ints[iT][iDT1+1], 3);
      if(integral>1.0e40) {
        break;
      }
    }
    assert( std::isfinite(integral) );
    Is[iT]=integral * 4.0*M_PI/3.0;
  }

  std::vector<double> test(NTs);
  for(int iT=0; iT<NTs; iT++) test[iT] = std::pow( Is[iT]-0.34, 2 );
  const auto itr_min = std::min_element( test.begin(), test.end() );
  const int iTstar = std::distance(test.begin(), itr_min);
  Tstar = Ts[iTstar];

  alpha = get_alpha(Tstar, Tc, MB, f_DV, f_S3 );
  vJ = getv(alpha);
  beta_tilde = Tstar * f_S3.dT( Tstar-Tc, MB );
}






// void get_gw_params_o( double& alpha,
//                     double& vJ,
//                     double& beta_tilde,
//                     double& Tstar,
//                     double& Tc,
//                     const F_Tc& f_Tc,
//                     const F_DeltaV& f_DV,
//                     const F_S3hat& f_S3,
//                     const double MB,
//                     const double MP,
//                     const int NTs = 100, // -> 400
//                     const double DTc = 0.0004 // # from DVPrime
//                     ){
//   Tc = f_Tc(MB);

//   std::vector<double> Ts(NTs);
//   std::vector<double> TmTcs(NTs);
//   const double dT = DTc/NTs;
//   for(int i=0; i<NTs; i++) {
//     Ts[i] = Tc-DTc + i*dT;
//     TmTcs[i] = -DTc + i*dT;
//   }

//   const F_Gammahat f_Gammahat = F_Gammahat( f_S3 );

//   std::vector<double> GamPrimes(NTs), DGamPrimes(NTs);
//   for(int i=0; i<NTs; i++) {
//     GamPrimes[i] = f_Gammahat( TmTcs[i], MB );
//     DGamPrimes[i] = f_Gammahat.dT( TmTcs[i], MB );
//   }


//   std::vector<double> alphas(NTs), vJs(NTs);
//   std::vector<double> Hsqs(NTs), Hinvs(NTs);
// #ifdef _OPENMP
// #pragma omp parallel for num_threads(nparallel)
// #endif
//   for(int i=0; i<NTs; i++) {
//     Hsqs[i] = std::pow(Ts[i],4)/(3.0*MP*MP) * ( M_PI*M_PI/30.0 * gstar(Ts[i]) );
//     Hinvs[i] = 1.0/std::sqrt(Hsqs[i]);

//     alphas[i] = get_alpha(Ts[i], Tc, MB, f_DV, f_S3 );
//     vJs[i] = getv(alphas[i]);
//   }

//   std::vector<std::vector<double>> T_T1_Ints(NTs); // [iT][iDT1] // 0 <= DT1 < Tc-T
//   for(int iT=0; iT<NTs; iT++) { // 0<=T<Tc
//     std::vector<double> Ints(NTs-iT); // = 0.0;
// #ifdef _OPENMP
// #pragma omp parallel for num_threads(nparallel)
// #endif
//     for(int iDT1=0; iDT1<NTs-iT; iDT1++){ // 0 <= DT1 < Tc-T
//       double Int = 0.0;
//       for(int iT2=iT; iT2<iT+iDT1; iT2++) Int += dT * Hinvs[iT2] * vJs[iT2];
//       Ints[iDT1]=Int;
//     }
//     T_T1_Ints[iT]=Ints;
//   }

//   std::vector<double> Is(NTs);
// #ifdef _OPENMP
// #pragma omp parallel for num_threads(nparallel)
// #endif
//   for(int iT=0; iT<NTs; iT++) {
//     double integral = 0.0;
//     for(int iDT1=0; iDT1<NTs-iT; iDT1++) {
//       const int iT1prime = iT+iDT1;
//       integral += dT * Hinvs[iT1prime] * GamPrimes[iT1prime] * std::pow(T_T1_Ints[iT][iDT1], 3);
//     }
//     assert( std::isfinite(integral) );
//     Is[iT]=integral * 4.0*M_PI/3.0;
//   }

//   std::vector<double> test(NTs);
//   for(int iT=0; iT<NTs; iT++) test[iT] = std::pow( Is[iT]-0.34, 2 );
//   const auto itr_min = std::min_element( test.begin(), test.end() );
//   const int iTstar = std::distance(test.begin(), itr_min);
//   Tstar = Ts[iTstar];

//   alpha = get_alpha(Tstar, Tc, MB, f_DV, f_S3 );
//   vJ = getv(alpha);
//   beta_tilde = Tstar * f_S3.dT( Tstar-Tc, MB );
// }





// double HinGeV(const double TinGeV){
//   return 1.0/(std::sqrt(3.0)*MP_GeV) * std::sqrt( M_PI*M_PI/30.0 * gstar(TinGeV) );
// }


// void get_gw_params_n( double& alpha,
//                       double& vJ,
//                       double& beta_tilde,
//                       double& Tstar,
//                       double& Tc,
//                       const F_Tc& f_Tc,
//                       const F_DeltaV& f_DV,
//                       const F_S3hat& f_S3,
//                       const double MB,
//                       const double MP,
//                       const int NTs = 100, // -> 400
//                       const double DTc = 0.0004 // # from DVPrime
//                       ){
//   Tc = f_Tc(MB);

//   // std::vector<double> Ts(NTs);
//   // std::vector<double> TmTcs(NTs);
//   // const double dT = DTc/NTs;
//   // for(int i=0; i<NTs; i++) {
//   //   Ts[i] = Tc-DTc + i*dT;
//   //   TmTcs[i] = -DTc + i*dT;
//   // }
//   const F_Gammahat f_Gammahat = F_Gammahat( f_S3 );

//   std::vector<double> GamPrimes(NTs), DGamPrimes(NTs);
//   // for(int i=0; i<NTs; i++) {
//   //   GamPrimes[i] = f_Gammahat( TmTcs[i], MB );
//   //   DGamPrimes[i] = f_Gammahat.dT( TmTcs[i], MB );
//   // }


//   std::vector<double> alphas(NTs), vJs(NTs);
//   std::vector<double> Hsqs(NTs), Hinvs(NTs);
// #ifdef _OPENMP
// #pragma omp parallel for num_threads(nparallel)
// #endif
//   for(int i=0; i<NTs; i++) {
//     Hsqs[i] = std::pow(Ts[i],4)/(3.0*MP*MP) * ( M_PI*M_PI/30.0 * gstar(Ts[i]) );
//     Hinvs[i] = 1.0/std::sqrt(Hsqs[i]);

//     alphas[i] = get_alpha(Ts[i], Tc, MB, f_DV, f_S3 );
//     vJs[i] = getv(alphas[i]);
//   }

//   std::vector<std::vector<double>> T_T1_Ints(NTs); // [iT][iDT1] // 0 <= DT1 < Tc-T
//   for(int iT=0; iT<NTs; iT++) { // 0<=T<Tc
//     // std::vector<double> Ints(NTs-iT); // = 0.0;
//     // for(int iDT1=0; iDT1<NTs-iT; iDT1++){ // 0 <= DT1 < Tc-T
//     for(int iT2=iT; iT2<iT+iDT1; iT2++) Int += dT * Hinvs[iT2] * vJs[iT2];
//     Ints[iDT1]=Int;
//     }
//     T_T1_Ints[iT]=Ints;
//   }

//   std::vector<double> Is(NTs);
// #ifdef _OPENMP
// #pragma omp parallel for num_threads(nparallel)
// #endif
//   for(int iT=0; iT<NTs; iT++) {
//     double integral = 0.0;
//     for(int iDT1=0; iDT1<NTs-iT; iDT1++) {
//       const int iT1prime = iT+iDT1;
//       integral += dT * Hinvs[iT1prime] * GamPrimes[iT1prime] * std::pow(T_T1_Ints[iT][iDT1], 3);
//     }
//     assert( std::isfinite(integral) );
//     Is[iT]=integral * 4.0*M_PI/3.0;
//   }

//   std::vector<double> test(NTs);
//   for(int iT=0; iT<NTs; iT++) test[iT] = std::pow( Is[iT]-0.34, 2 );
//   const auto itr_min = std::min_element( test.begin(), test.end() );
//   const int iTstar = std::distance(test.begin(), itr_min);
//   Tstar = Ts[iTstar];

//   alpha = get_alpha(Tstar, Tc, MB, f_DV, f_S3 );
//   vJ = getv(alpha);
//   beta_tilde = Tstar * f_S3.dT( Tstar-Tc, MB );
// }








void get_freq_amplitude( double& fpeak, double& hsq_Omega_peak,
                         const double alpha,
                         const double vJ,
                         const double beta_tilde,
                         const double Tstar,
                         // const double Tc,
                         const double sqrt_t0_inv_GeV){
  const double kappa_v = std::sqrt(alpha)/(0.135+std::sqrt(0.98+alpha));
  const double Ubarfsq = 3.0/4.0 * alpha/(1.0+alpha) * kappa_v;
  const double tau_sw = 1 - 1.0 / std::sqrt( 1.0 + 2.0 * std::pow(8.0*M_PI, 1.0/3.0) * vJ / ( beta_tilde*std::sqrt(Ubarfsq) ) );
  const double kappa_sw = std::sqrt(tau_sw)*kappa_v;
  const double Tstar_GeV = Tstar*sqrt_t0_inv_GeV;

  fpeak = 1.9e-5 * std::pow(gstar(Tstar_GeV)/100.,1.0/6.0) * (Tstar_GeV/100.) * (beta_tilde / vJ);
  hsq_Omega_peak = 2.65e-6 * (vJ/beta_tilde) * std::pow(kappa_sw*alpha/(1.0+alpha),2) * std::pow(100./gstar(Tstar_GeV),1.0/3.0) * std::pow(Om_HSDM(Tstar_GeV),2);
}
