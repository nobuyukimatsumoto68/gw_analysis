#pragma once







std::vector<double> find_all_roots( const std::function<double(const double)>& f,
                                    const std::function<double(const double)>& f_prime,
                                    const double xmin, const double xmax,
                                    const double resolution,
                                    const int iter_max = 100000,
                                    const double tol=1.0e-12 ){
  std::vector<double> roots;
  for(double x0=xmin+2*resolution; x0<xmax-2*resolution; x0+=resolution){
    const bool is_pos1 = f(x0) > 0.;
    const bool is_pos2 = f(x0+resolution) > 0.;
    if( is_pos1 != is_pos2 ){
      double root = x0;
      for(int i=0; i<iter_max; i++){
        const double root_old = root;
        root = root - f(root)/f_prime(root);
        root = std::min( std::max( root, x0 ), x0+resolution );
        if( std::abs(root-root_old)<tol ) break;
      }
      // std::cout << "debug root = " << root << " f = " << f(root) << std::endl;
      roots.push_back(root);
    }
  }
  return roots;
}








double force(const double t,
             const double p,
             const double q,
             const std::function<double(const double)>& dV
             ){
  // std::cout << "debug. q = " << q << std::endl
  //           << " dV = " << dV(q) << std::endl;
  return -2.0/t * p + 0.5*dV(q);
}

void leapfrog(const double t,
              double& p,
              double& q,
              const double tau,
              const std::function<double(const double)>& dV
              ){
  double p_half = p + 0.5 * tau * force(t+0.5*tau, p, q, dV);
  double p_half_old = p_half;
  for(int i=0; i<2; i++){
    p_half = p + 0.5 * tau * force(t+0.5*tau, p_half, q, dV);
    if( std::abs(p_half-p_half_old)<1.0e-10) break;
    p_half_old = p_half;
  }

  q = q + tau * p_half;
  p = p_half + 0.5 * tau * force(t+0.5*tau, p_half, q, dV);
}


void integrate( std::vector<double>& ts,
                std::vector<double>& qs,
                const double p_init,
                const double q_init,
                const double xA,
                const double xB,
                const std::function<double(const double)>& dV,
                const double tmax=250.,
                const double tau=1.0e-3
                ){
  ts.clear();
  qs.clear();

  const int nsteps = static_cast<int>(std::round(tmax/tau));
  // std::cout << "debug. nsteps = " << nsteps
  //           << " tmax = " << tmax
  //           << " tau = " << tau << std::endl;
  double p = p_init;
  double q = q_init;
  for(int i=0; i<nsteps; i++){
    const double t = i*tau;
    leapfrog(t, p, q, tau, dV);
    ts.push_back(t);
    qs.push_back(q);
    if( q<xA-tau || q>xB+tau ) break;
  }
}


double cost( std::vector<double>& ts,
             std::vector<double>& qs,
             const double q_init,
             const double xA,
             const double xB,
             const std::function<double(const double)>& dV,
             const double tmax=250.,
             const double tau=1.0e-3
             ){
  const double p_init = 0.0;
  integrate( ts, qs, p_init, q_init, xA, xB, dV, tmax, tau );
  // std::cout << "# debug. ts[-1] = " << ts[ts.size()-1] << ", qs[-1] = " << qs[qs.size()-1] << std::endl;
  return qs[qs.size()-1] - xB;
}


double search_root( std::vector<double>& ts,
                    std::vector<double>& qs,
                    double& dq_init,
                    double delta,
                    const double xA,
                    const double xB,
                    const std::function<double(const double)>& dV,
                    const double tmax=250.,
                    const double tau=1.0e-3,
                    const int iter_max=1000,
                    const double tol=1.0e-4,
                    const double tol2=1.0e-17
                    ){
  assert( tau*tau<tol );
  double c = cost( ts, qs, xA+dq_init, xA, xB, dV, tmax, tau );
  double c_old = c;

  // std::cout << "# debug. pt2" << std::endl;
  // std::cout << "pt1 dq_init = " << dq_init << std::endl;

  for(int i=0; i<iter_max; i++){
    if( c*c_old < 0. ) {
      // std::cout << "# debug. pt1 " << std::endl;
      delta *= 0.5;
    }
    c_old = c;
    // else{
    if(c<0.) {
      while(dq_init-delta < 0.) {
        delta *= 0.5;
        // if(delta<tol2) {
        //   std::cout << "# !!! delta too small break." << std::endl;
        //   break;
        // }
      }
      dq_init -= delta;
      // q_init = std::max(q_init, xA);
      // if( std::abs(q_init-xA)<tol2 ) {
      //   std::cout << "# !!! q_init too close to xA." << std::endl;
      //   break;
      // }
    }
    else dq_init += delta;
    // std::cout << "pt3 dq_init = " << dq_init << std::endl;
    // }
    c = cost( ts, qs, xA+dq_init, xA, xB, dV, tmax, tau );
    std::cout << "# debug. c = " << c
              << "# debug. c+xB = " << c+xB
      // << "# debug. c_old = " << c_old
              // << "# bool = " << (c*c_old < 0)
              << " dq_init = " << dq_init
              << " delta = " << delta << std::endl;
    // STRICT criterion (ORIGINAL eps2, the CORRECT bounce): accept only when the field rides
    // ALL the way to tmax staying in-bounds, i.e. it asymptotes to xB with q'->0. This is the
    // true critical bounce; a completed-early trajectory is a marginal OVERSHOOT (q'!=0 at xB)
    // whose action is biased high, so it must NOT be accepted. Intermediate betas can reach this
    // (window > machine precision); only the genuine near-Tc machine-precision betas cannot.
    if( std::abs(c)<tol && std::abs(ts[ts.size()-1]-tmax)<tol && ts.size()==static_cast<int>(std::round(tmax/tau)) ) {
      std::cout << "# debug. break." << std::endl;
      break;
    }
    // RELAXED break (WRONG -- accepts marginal overshoots, biased S3). Kept for reference:
    // if( std::abs(c)<tol && std::abs(qs[qs.size()-1]-qs[0]) > 0.4*std::abs(xB-xA) ) {
    //   break;
    // }
    if(delta<tol2) {
      std::cout << "# !!! delta too small break." << std::endl;
      break;
    }
    if( std::abs(c-c_old)<tol2 ){
      std::cout << "# !!! no cost difference break" << std::endl;
      break;
    }
  }

  return c;
}



void get_Gamma( Eigen::ArrayXXd& Gamma_list,
                const int nbeta_meas,
                const int nptsx,
                const int nptsy,
                const int irow1,
                const std::string& basedir,
                const std::string& mass,
                const int Ns,
                const int Nt,
                const std::string& desc="nojk" // or jk_{jdrop}_{nbin}_{ibin}
                ){
  Eigen::ArrayXXd hist_list(nbeta_meas, nptsx);
  const int ipty = irow1;
  for(int iptx=0; iptx<nptsx; iptx++){
    Eigen::ArrayXd yy1, yy2;
    {
      const std::string filename1 = basedir + "/m"+mass+"avghist_ibx"+std::to_string(iptx)+"_iby"+std::to_string(ipty)+"_"+desc+"meas.bin";
      const HighFive::File f1(filename1.c_str(), HighFive::File::ReadOnly);

      std::vector<double> tmp;
      Eigen::ArrayXd f, fP;
      tmp = f1.getDataSet("f").read<std::vector<double>>();
      f = Eigen::Map<Eigen::ArrayXd>(tmp.data(), tmp.size());
      tmp = f1.getDataSet("fP").read<std::vector<double>>();
      fP = Eigen::Map<Eigen::ArrayXd>(tmp.data(), tmp.size());
      yy1 = (f-fP).exp();
    }

    {
      const std::string filename2 = basedir + "/m"+mass+"avghist_ibx"+std::to_string(iptx)+"_iby"+std::to_string(nptsy-ipty-1)+"_"+desc+"meas.bin";
      const HighFive::File f2(filename2.c_str(), HighFive::File::ReadOnly);

      std::vector<double> tmp;
      Eigen::ArrayXd f, fP;
      tmp = f2.getDataSet("f").read<std::vector<double>>();
      f = Eigen::Map<Eigen::ArrayXd>(tmp.data(), tmp.size());
      tmp = f2.getDataSet("fP").read<std::vector<double>>();
      fP = Eigen::Map<Eigen::ArrayXd>(tmp.data(), tmp.size());
      yy2 = (f-fP).exp();
    }

    hist_list.block( 0, iptx, nbeta_meas, 1 ) = 0.5*(yy1+yy2);
  }

  Gamma_list.resize(nbeta_meas, nptsx);
  Gamma_list = -std::pow(1.0*Nt/Ns,3) * hist_list.log();
}



void get_local_minima( double& xA, double& xB,
                       const std::vector<double>& xpts,
                       const double deltax,
                       const std::function<double(const double)>& Veff_prime,
                       const std::function<double(const double)>& Veff_prime2
                       ){
  const double xmin= *std::min_element(xpts.begin(), xpts.end()); // *std::min(xpts); // -0.1;
  const double xmax= *std::max_element(xpts.begin(), xpts.end()); // *std::min(xpts); // -0.1;
  const double resolution = 0.01*deltax;
  // const std::vector<double> roots = find_all_roots( Veff_prime, Veff_prime2, xmin+1, xmax-1, resolution ); // avoid the edges
  const std::vector<double> roots = find_all_roots( Veff_prime, Veff_prime2, xmin, xmax, resolution ); // avoid the edges

  int iA=0;
  int iB=0;

  // if( Veff_prime(xpts[0])<0. ) iA=1;
  // if( Veff_prime(xpts[-1])<0. ) iB=1;

  for(auto itr = roots.begin(); itr != roots.end(); itr++) {
    // std::cout << "debug. *itr = " << *itr << std::endl;
    if( Veff_prime2(*itr)>=0.0 ) {
      xA = *itr;
      break;
    }
  }
  for(auto itr = roots.rbegin(); itr != roots.rend(); itr++) {
    // std::cout << "debug. *itr = " << *itr << std::endl;
    if( Veff_prime2(*itr)>=0.0 ) {
      xB = *itr;
      break;
    }
  }
  // xA = roots[iA];
  // xB = roots[roots.size()-1-iB];
  // std::cout << "debug. root prime2: " << std::endl;
  // for(auto elem : roots) std::cerr << elem << " " << Veff_prime(elem) << " " << Veff_prime2(elem) << std::endl;
  // std::cerr << std::endl;
  // std::cout << "debug. iA, iB " << iA << " " << iB << std::endl;
  // assert( Veff_prime2(xA)>-eps );
  // assert( Veff_prime2(xB)>-eps );
}





void write(const std::string& filename,
           const std::vector<double>& ts,
           const std::vector<double>& qs,
           const std::string& comment=""){

  std::ofstream ofs(filename);
  ofs << std::scientific << std::setprecision(25);

  assert( ts.size()==qs.size() );

  ofs << "# " << comment << std::endl;

  for(int i=0; i<ts.size(); i++) {
    ofs << std::setw(50) << ts[i] << "\t"
        << std::setw(50) << qs[i] << std::endl;
  }
}




