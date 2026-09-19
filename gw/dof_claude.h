const int nLightF = 6+4+4 + 36;
const int nLightG = 2 + 16;

const double TQCD = 0.155; //1402.5175

const double Mc = 1.273;
const int nc = 12;

const double Mtau = 1.77693;
const int ntau = 4;

const double Mb = 4.183;
const int nb = 12;

const double MWZ = 90.;
const int nWZ = 9;

const double MH = 125;
const int nH = 1;

const double Mt = 172.56;
const int nt = 12;


const int Nc=4;
const int Nf=1;
const int Ns=32;
const int Nt=8;

// CORRECTED HSDM dof: a Dirac fermion in the fundamental of SU(Nc) has
// 4*Nc*Nf real dof (particle+antiparticle \times 2 spin \times Nc \times Nf).
// Old value 33.5 used 7/8*4*Nf*Nf = 3.5 for the fermion term (wrong: Nf*Nf);
// correct fermion term is 7/8*4*Nc*Nf = 14, giving 2*(Nc*Nc-1) + 14 = 44.
const double gstar_HSDM = 44.0; // was 33.5; = 2*(Nc*Nc-1)+7.0/8.0*4.0*Nc*Nf
// const double gstar_SM = 106.75;
// const double gstar = gstar_SM + gstar_HSDM;
// const double Om_HSDM = gstar_HSDM / (gstar_HSDM+gstar_SM);
const double MP_GeV = 2.435e18;// 1.220890e19; //GeV https://physics.nist.gov/cgi-bin/cuu/Value?plkmc2gev


double gstar_SM(const double T){ // in GeV
  // assert( T>=TQCD );
  double res = (7.0/8.0)*nLightF + nLightG;
  if(T>Mc) res += (7.0/8.0) * nc;
  if(T>Mtau) res += (7.0/8.0) * ntau;
  if(T>Mb) res += (7.0/8.0) * nb;
  if(T>MWZ) res += nWZ;
  if(T>MH) res += nH;
  if(T>Mt) res += (7.0/8.0) * nt;
  return res;
}

inline double gstar(const double T){
  return gstar_SM(T) + gstar_HSDM;
}

inline double Om_HSDM(const double T){
  return gstar_HSDM / gstar(T);
}

inline double rhoRprime( const double T ){
  return M_PI*M_PI/30.0 * gstar(T);
}

// inline double rhoRprime( const double T ){
//   return M_PI*M_PI/30.0 * geff(T);
// }
