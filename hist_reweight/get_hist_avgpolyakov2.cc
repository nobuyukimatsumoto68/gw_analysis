// #include "reweight.h"
#include "reweight2.h"

#include <regex>


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

  int type=1;
  if(basedir2==basedir) {
    if(mass=="0p1000" || mass=="0p4000") type=2;
    else if(mass=="0p0500" || mass=="0p0100" || mass=="0p2000" || mass=="0p3000") type=4;
    else if(mass=="0") type=6;
  }
  else if(mass=="0") type = 0;

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

      std::string pathO3 = basedir2+obs_id+"avgPr_hist"+std::to_string(conf)+".bin";
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
          hist[ibx][iby] = regularized_delta( obsR.O, obsI.O,
                                              xs[ibx], ys[iby],
                                              delta_x, delta_y );
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
