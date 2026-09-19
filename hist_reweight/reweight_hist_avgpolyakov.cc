// #include "reweight.h"
#include "reweight2.h"

#include <regex>

// #define NOSIGN

int main(int argc, char **argv) {
  Grid_init(&argc, &argv);
  int threads = GridThread::GetThreads();
  std::cout << GridLogMessage << "Grid is setup to use " << threads << " threads" << std::endl;
  // omp_set_num_threads(threads);
  // std::cout << "max threads = " << omp_get_max_threads() << std::endl;
  // std::cout << "num threads = " << omp_get_num_threads() << std::endl;


  // -------------------------------------
  // for reading data
  const int conf_min0=atoi(argv[1]);
  const std::string base_dir(argv[2]);
  const std::string basedir(argv[3]);
  const std::string basedir2c(argv[4]);
  const std::string basedir2h(argv[5]);
  // const std::string basedir2(argv[4]);
  const std::string mass(argv[6]);
  const int conf_max0=atoi(argv[7]);
  const int nbin_global=atoi(argv[8]);
  int nbeta=atoi(argv[9]);
  int runtype=atoi(argv[10]);
  const int nbins_histo_x=atoi(argv[11]);
  double min_x = atof(argv[12]);
  double max_x = atof(argv[13]);
  const int nbins_histo_y=atoi(argv[14]);
  double min_y = atof(argv[15]);
  double max_y = atof(argv[16]);
  const double beta_meas_min=atof(argv[17]);
  const double beta_meas_max=atof(argv[18]);
  const int nbeta_meas=atoi(argv[19]);


  std::cout << "nbeta = " << nbeta << std::endl;
  std::cout << "basedir = " << basedir << std::endl;
  std::cout << "nbins_histo_x = " << nbins_histo_x << std::endl;
  std::cout << "min_x = " << min_x << std::endl;
  std::cout << "max_x = " << max_x << std::endl;
  std::cout << "nbins_histo_y = " << nbins_histo_y << std::endl;
  std::cout << "min_y = " << min_y << std::endl;
  std::cout << "max_y = " << max_y << std::endl;

  // int type=1;
  // if(basedir2c==basedir2h) {
  //   if(mass=="0p1000") type=2;
  //   else if(mass=="0p0500" || mass=="0p0100" || mass=="0p2000") type=4;
  // }
  int type=1;
  if(basedir2c==basedir2h) {
    if(mass=="0p1000" || mass=="0p4000") type=2;
    else if(mass=="0p0500" || mass=="0p0100" || mass=="0p2000" || mass=="0p3000") type=4;
    else if(mass=="0") type=6;
  }
  else if(mass=="0") type = 0;

  if(runtype>=2) {
    type+=1;
    runtype -= 2;
  }


  // -----------------------

  std::vector<std::string> betas;
  std::vector<double> betas_double;
  {
    for(int i=20; i<20+nbeta; i++) {
      std::string str(argv[i]);
      betas.push_back(str);

      std::regex to_replace("p");
      str = std::regex_replace(str, to_replace, ".");
      std::cout << "str replaced = " << str << std::endl;

      betas_double.push_back(std::stod(str));
    }
  }

  using Impl = PeriodicGimplR;


  // ################################################################ //
  std::cout << GridLogMessage << "run histogram part" << std::endl;

  char id[200];
  std::sprintf(id, "m%s", mass.data());

  // no openmp loop so far
  {

    std::vector<std::vector<std::vector<EnsembleOfObs>>> array_of_hist_ens( nbins_histo_x,
                                                                            std::vector<std::vector<EnsembleOfObs>>(nbins_histo_y,
                                                                                                                    std::vector<EnsembleOfObs>(nbeta)
                                                                                                                    )
                                                                            );

// #pragma omp parallel for
    for(int j=0; j<nbeta; j++){
      const std::string obs_id = get_configname(betas[j], mass, type);

      for(std::string basedir2 : std::vector<std::string>{basedir2c, basedir2h} ){
        for(int conf=conf_min0; conf<conf_max0; conf++){

          std::string pathO; //  = basedir2+obs_id+std::to_string(conf)+".bin";
          if(type>1) pathO = basedir2c+obs_id+std::to_string(conf)+".h5";
          else pathO = basedir2c+obs_id+std::to_string(conf)+".bin";

          std::string pathO3 = basedir2+obs_id+"avgPr_hist"+std::to_string(conf)+".bin";
          if( !std::filesystem::exists( pathO3 ) || !std::filesystem::exists( pathO ) ) {
            if(conf%conf_min0==0 && conf <= 10*conf_min0){
              std::cout << "debug. pathO = " << pathO << std::endl;
              std::cout << "debug. pathO3 = " << pathO3 << std::endl;
            }
            continue; // assert(false);
          }

          HistObs hist;
          Obs obs;
          {
            XmlReader WR( pathO3 );
            // jdrop
            read(WR, "hist", hist.vO );
          }
#pragma omp critical
          {
            std::unique_ptr<Hdf5Reader> WR;
            WR = std::make_unique<Hdf5Reader>( pathO );
            // jdrop
            read(*WR, "obs", obs );
          }

          // push_back
          for(int ibx=0; ibx<nbins_histo_x; ibx++){ // histogram binning for
            for(int iby=0; iby<nbins_histo_y; iby++){ // histogram binning for
              array_of_hist_ens[ibx][iby][j].Os.push_back( hist.vO[ibx][iby] );
              array_of_hist_ens[ibx][iby][j].energies.push_back( obs.energy );
            }}
        } // conf
      } // cold, hot
      for(int ibx=0; ibx<nbins_histo_x; ibx++){ // histogram binning for
        for(int iby=0; iby<nbins_histo_y; iby++){ // histogram binning for
          EnsembleOfObs& ens = array_of_hist_ens[ibx][iby][j]; // by ref
          assert( ens.size()!=0 );
          ens.nbin = nbin_global;
          ens.binsize = ens.size()/nbin_global;
          ens.nconf = ens.nbin*ens.binsize;
          { // trimming
            const int start = ens.nconf;
            ens.energies.erase( ens.energies.begin()+start, ens.energies.end() );
            ens.Os.erase( ens.Os.begin()+start, ens.Os.end() );
            // ens.Osqs.erase( ens.Osqs.begin()+start, ensR.Osqs.end() );
          }
          assert( ens.size()==ens.nconf );
        }}
    } // for j






    // jackknife below

    std::cout << "debug. hist" << std::endl;
    const int nbin=nbin_global; // ensembles0[jdrop].nbin;

#pragma omp parallel for num_threads(threads) collapse(2)
    for(int ibx=0; ibx<nbins_histo_x; ibx++){ // histogram binning for
      for(int iby=0; iby<nbins_histo_y; iby++){ // histogram binning for

        std::cout << GridLogMessage
                  << "ibx = " << ibx
                  << " iby = " << iby << std::endl;

        for(int jdrop=0; jdrop<nbeta; jdrop++){
          for(int ibin=0; ibin<nbin; ibin++){
            std::vector<EnsembleOfObs> ensembles;
            for(int i=0; i<nbeta; i++) ensembles.push_back( array_of_hist_ens[ibx][iby][i] ); // by copy
            ensembles[jdrop].jackknife_drop( nbin, ibin );

            // out path
            std::string id_str = id;
            // jdrop
            id_str = id_str+"avghist_ibx"+std::to_string(ibx)+"_iby"+std::to_string(iby);
            id_str = id_str+"_jk_"+std::to_string(jdrop)+"_"+std::to_string(nbin)+"_"+std::to_string(ibin);
            std::string path_res = basedir+id_str+"meas.bin";
            if( std::filesystem::exists( path_res ) && (runtype==0) ) continue; // assert(false);

            // ReweightedObs meas( betas_double.front(), betas_double.back(), nbeta*100, ensembles );
            ReweightedObs meas( beta_meas_min, beta_meas_max, nbeta_meas, ensembles );
            const int nmeas=meas.nmeas;

            // read fs
            FreeEnergies fs(nbeta);
            {
              // in path
              std::string id_str = id;
              // jdrop
              id_str = id_str+"_jk_"+std::to_string(jdrop)+"_"+std::to_string(nbin)+"_"+std::to_string(ibin);
              // id_str = id_str+"_nojk";
#pragma omp critical
              {
                std::unique_ptr<Hdf5Reader> WR;
                WR = std::make_unique<Hdf5Reader>( basedir+id_str+"fs.bin" );
                read(*WR, "f", fs.f );
              }
            }

            run_reweighting( betas_double, fs, meas, false );

#pragma omp critical
            {
              std::unique_ptr<Hdf5Writer> WR;
              WR = std::make_unique<Hdf5Writer>( path_res );
              write(*WR, "beta", meas.betas );
              write(*WR, "f", meas.fs );
              write(*WR, "fP", meas.fPs );
            }

          }} // end jackknife loop

      }} // end for ib
  }


  Grid_finalize();
}
