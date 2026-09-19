// #include "reweight.h"
#include "reweight2.h"

#include <regex>

// ---------------------------------------------------------------------------
// v16 variant of get_hist_avgpolyakov2.cc (32c redo).
// Reverts the sig0p5 halving: the Gaussian delta-regularization width is set back
// to the FULL bin spacing (sigma = binsize, epsilon-factor 1.0, = original / 24c),
// so delta_x_smear/delta_y_smear = delta_x/delta_y. The bias reduction is instead
// obtained by DOUBLING the ImL bins (nbins_histo_y = 80 -> delta_y = 0.005), passed
// as a run-script arg (no code change for the binning). Rationale: the absolute
// sigma sets the kernel bias (~sigma^2); the sigma/binsize RATIO sets the noise,
// and ratio 1 (sigma=binsize) is the smoothest -> the same sigma_y=0.005 that
// sig0p5 reached by halving (noisy, spurious ReL wiggles) is reached here by a
// finer bin at full sigma (smooth). See veff_32c_v16_impl_plan_claude.md and the
// b=11.03 scan hist_Bmn_Nh_coarse32_claude.py. Only the 32c set is redone.
// The op-in / hist-out split (read op from basedir2=32_v9, write avgPr_hist to
// basedir=32_v16) is kept from the sig0p5 copy. avgPr_hist output format unchanged,
// so the existing realaxis reweight binary reads these unchanged.
// ---------------------------------------------------------------------------


int main(int argc, char **argv) {
  Grid_init(&argc, &argv);
  int threads = GridThread::GetThreads();
  std::cout << GridLogMessage << "Grid is setup to use " << threads << " threads" << std::endl;

  // -------------------------------------
  // for reading data
  const int conf_min0=atoi(argv[1]);
  const std::string base_dir(argv[2]);
  const std::string basedir(argv[3]);
  const std::string basedir2(argv[4]);
  // const std::string basedir2h(argv[5]);
  const std::string mass(argv[5]);
  const int interval=atoi(argv[6]); // !!!!!!!!! @@@@@@@@@@@
  const int conf_max0=atoi(argv[7]);
  // const int binsize=atoi(argv[9]);
  const int nbin_global=atoi(argv[8]); // dummy
  int nbeta=atoi(argv[9]);
  int runtype=atoi(argv[10]);
  // obsinfo = argv[12]; // dummy
  const int nbins_histo_x=atoi(argv[11]);
  double min_x = atof(argv[12]);
  double max_x = atof(argv[13]);
  const int nbins_histo_y=atoi(argv[14]);
  double min_y = atof(argv[15]);
  double max_y = atof(argv[16]);

  std::cout << "nbeta = " << nbeta << std::endl;
  std::cout << "basedir = " << basedir << std::endl;
  std::cout << "nbins_histo_x = " << nbins_histo_x << std::endl;
  std::cout << "min_x = " << min_x << std::endl;
  std::cout << "max_x = " << max_x << std::endl;
  std::cout << "nbins_histo_y = " << nbins_histo_y << std::endl;
  std::cout << "min_y = " << min_y << std::endl;
  std::cout << "max_y = " << max_y << std::endl;

  // std::string obsinfoR="flow_polyakov_re";
  // std::string obsinfoI="flow_polyakov_im";

  // sig0p5: the op input (basedir2=32_v9) and the avgPr_hist output (basedir=32_v15) now
  // live in DIFFERENT dirs, so the original `basedir2==basedir` gate can no longer select
  // the production op naming scheme. Compute `type` directly from `mass` (the production
  // basedir2==basedir branch), unconditional of the dir names.
  // ORIGINAL (gated on basedir2==basedir):
  // int type=1;
  // if(basedir2==basedir) {
  //   if(mass=="0p1000" || mass=="0p4000") type=2;
  //   else if(mass=="0p0500" || mass=="0p0100" || mass=="0p2000" || mass=="0p3000") type=4;
  //   else if(mass=="0") type=6;
  // }
  // else if(mass=="0") type = 0;
  int type=1;
  if(mass=="0p1000" || mass=="0p4000") type=2;
  else if(mass=="0p0500" || mass=="0p0100" || mass=="0p2000" || mass=="0p3000") type=4;
  else if(mass=="0") type=6;

  if(runtype>=2) {
    type+=1;
    runtype -= 2;
  }


  std::vector<double> ainv_coeffs;
  std::string mass2;
  {
    std::ifstream file;

    if(mass=="0p1000" || mass=="0.1") mass2="0.1";
    else if(mass=="0p2000" || mass=="0.2") mass2="0.2";
    else if(mass=="0p3000" || mass=="0.3") mass2="0.3";
    else if(mass=="0p4000" || mass=="0.4") mass2="0.4";
    else assert(false);

    // std::string fname = base_dir + "coeffs_m"+mass2+"_t0hat_c0_c1beta.dat";
    std::string fname = base_dir + "coeffs_ainv_beta_mqhat.dat";
    std::cout << "debug. fname = " << fname << std::endl;
    file.open( fname );

    std::string str;
    while (std::getline(file, str)){
      std::istringstream iss(str);
      double v;
      iss >> v;
      ainv_coeffs.push_back( v );
    }
    // assert(ainv_coeffs.size()==2);
    assert(ainv_coeffs.size()==6);
  }
  // std::cout << "debug: " << ainv_coeffs[0] << " " << ainv_coeffs[1] << std::endl;


  std::vector<std::string> betas;
  std::vector<double> betas_double;
  {
    for(int i=17; i<17+nbeta; i++) {
      std::string str(argv[i]);
      betas.push_back(str);

      std::regex to_replace("p");
      str = std::regex_replace(str, to_replace, ".");
      std::cout << "str replaced = " << str << std::endl;

      // std::replace( str, "p", "." );
      betas_double.push_back(std::stod(str));
    }
  }


  // -----------------------




  std::vector<double> xs(nbins_histo_x);
  const Real delta_x = (max_x-min_x)/nbins_histo_x;
  for(int i=0; i<nbins_histo_x; i++) xs[i] = min_x + delta_x*(0.5+i);
  std::cout << "xs = ";
  for(auto elem : xs) std::cout << elem << " ";
  std::cout << std::endl;

  std::vector<double> ys(nbins_histo_y);
  const Real delta_y = (max_y-min_y)/nbins_histo_y;
  for(int i=0; i<nbins_histo_y; i++) ys[i] = min_y + delta_y*(0.5+i);
  std::cout << "ys = ";
  for(auto elem : ys) std::cout << elem << " ";
  std::cout << std::endl;

  // v16: smearing width = FULL bin spacing (sigma = binsize, epsilon-factor 1.0).
  // sig0p5 (reverted): const Real delta_x_smear = 0.5*delta_x;
  // sig0p5 (reverted): const Real delta_y_smear = 0.5*delta_y;
  const Real delta_x_smear = delta_x;
  const Real delta_y_smear = delta_y;
  std::cout << "v16: delta_x_smear = " << delta_x_smear
            << ", delta_y_smear = " << delta_y_smear << std::endl;



  std::cout << "conf_max0 = " << conf_max0 << std::endl;


  int counter=0;
  for(std::string beta : betas){
    std::cout << "beta = " << beta << std::endl;

    int conf_max=conf_min0;
    for(int conf=conf_min0; conf<conf_max0; conf+=interval){
      char f[200];
      std::string prefix = get_prefix(beta, mass);
      std::sprintf(f,
                   "%s.%d",
                   prefix.data(), conf);

      std::string pathO;
      if(type>1) pathO = basedir2+get_configname(beta, mass, type)+std::to_string(conf)+".h5";
      else pathO = basedir2+get_configname(beta, mass, type)+std::to_string(conf)+".bin";
      if( !std::filesystem::exists( pathO ) ) {
        std::cout << "missing pathO = " << pathO << std::endl;
        break;
      }
      conf_max = conf;
    }

    std::cout << "conf_max = " << conf_max << std::endl;

    for(int conf=conf_min0; conf<conf_max; conf+=interval){
      std::string obs_id=get_configname(beta, mass, type);

      // sig0p5: write the new (halved-sigma) avgPr_hist to basedir (32_v15), NOT basedir2
      // (32_v9). op is still READ from basedir2 below. -> shell sets basedir2=op-in (32_v9),
      // basedir=hist-out (32_v15); production 32_v9 avgPr_hist untouched.
      // ORIGINAL: std::string pathO3 = basedir2+obs_id+"avgPr_hist"+std::to_string(conf)+".bin";
      std::string pathO3 = basedir+obs_id+"avgPr_hist"+std::to_string(conf)+".bin";
      if( std::filesystem::exists( pathO3 ) && (runtype==0) ) {
        // std::cout << GridLogMessage << "skip." << std::endl;
        continue;
      }

      // Complex chcond;
      // std::vector<ComplexD> all_polyakov;
      std::string pathO = basedir2+obs_id+std::to_string(conf)+".bin";
      if(type>1) pathO = basedir2+obs_id+std::to_string(conf)+".h5";
      else pathO = basedir2+obs_id+std::to_string(conf)+".bin";
      if( !std::filesystem::exists( pathO ) ) continue; // assert(false);

      Obs obsR, obsI;
#pragma omp critical
      {
        std::unique_ptr<Hdf5Reader> WR;
        WR = std::make_unique<Hdf5Reader>( pathO );
        read(*WR, "obs", obsR );
        obsI = obsR;
        std::vector<ComplexD> all_polyakov;
        std::vector<RealD> all_T0;
        read(*WR, "polyakov", all_polyakov );
        read(*WR, "T0", all_T0 );

        assert( betas[counter]==beta );
        const ComplexD poly = interpolator( ainv_coeffs, betas_double[counter], std::stod(mass2),
                                            all_polyakov );
        obsR.O = real( poly ); // @@ !!
        obsI.O = imag( poly ); // @@ !!
        // obsR.O = real( all_polyakov.back() ); // @@ !!
        // obsI.O = imag( all_polyakov.back() ); // @@ !!
      }
      // std::cout << "debug. chcond = " << chcond << std::endl;

      // std::vector<double> hist( nbins_histo_x, 0.0 );
      // for(int ibx=0; ibx<nbins_histo_x; ibx++){ // histogram binning for
      //   hist[ibx] = regularized_delta( real(chcond), xs[ibx], delta_x );
      // } // end for ibx

      // {
      //   XmlWriter WR( pathO3 );
      //   write(WR, "hist", hist );
      //   write(WR, "xs", xs );
      // }


      std::vector<std::vector<double>> hist( nbins_histo_x,
                                             std::vector<double>(nbins_histo_y, 0.0));

      for(int ibx=0; ibx<nbins_histo_x; ibx++){ // histogram binning for
        for(int iby=0; iby<nbins_histo_y; iby++){ // histogram binning for
          // sig0p5: pass the halved smearing widths (NOT delta_x/delta_y).
          hist[ibx][iby] = regularized_delta( obsR.O, obsI.O,
                                              xs[ibx], ys[iby],
                                              delta_x_smear, delta_y_smear );
        } // end for iby
      } // end for ibx
      //obs.vO = hist;

      {
        XmlWriter WR( pathO3 );
        WR.scientificFormat( true ); // @@@ double check
        WR.setPrecision( std::numeric_limits<Real>::digits10 + 1 );
        write(WR, "hist", hist );
        write(WR, "xs", xs );
        write(WR, "ys", ys );
      }

      // {
      //   if (UGrid->IsBoss()){
      //     std::unique_ptr<Hdf5Writer> WR;
      //     WR = std::make_unique<Hdf5Writer>( pathO3 );
      //     write(*WR, "obs", obs );
      //     // write(*WR, "hist", hist );
      //   }
      // }
    } // end conf
    counter++;
  } // end beta



  // -----------------------


    Grid_finalize();
}
