This is a fork of the parnassus repository.

This fork handles data generated for the upcoming EIC.
This fork handles data generated using official ePIC simulation campaign files. 18 GeV on 275 GeV protons (highest beam energies) neutral current DIS reactions. The minQ2=100 and minQ2=1000 is used. Q2>=100 is used for most initial studies.
    
The data 
Truth particles: MCParticles branch. You can get the final state particles by masking with MCParticles.generatorStatus==1
Reconstructed DIS quantities: InclusiveKinematicsDA, InclusiveKinematicsElectron, InclusiveKinematicsESigma, InclusiveKinematicsJB, InclusiveKinematicsSigma branches. These are all different reconstruction techniques as seen in Table 1 in this paper: https://arxiv.org/pdf/2110.05505. The data for training initially uses ESigma, but the other reconstructed quantities commented out.

DIS kinematics list = ['x', 'Q2', 'W', 'y', 'nu']

dis_root_to_h5.py converts many root files /global/cfs/cdirs/m3246/eic/NC_DIS_18x275/ to hdf5 with the following structure:
'''
  reco_particle_list = [
      'ReconstructedParticles.energy',  # energy used for sorting (index 1)
      'ReconstructedParticles.momentum.x',
      'ReconstructedParticles.momentum.y',
      'ReconstructedParticles.momentum.z',
      'ReconstructedParticles.referencePoint.x',
      'ReconstructedParticles.referencePoint.y',
      'ReconstructedParticles.referencePoint.z',
      'ReconstructedParticles.charge',
      'ReconstructedParticles.mass',
      'ReconstructedParticles.PDG',
      'ReconstructedParticles.type',
  ]

  gen_particle_list = [
      'MCParticles.generatorStatus',  # generatorStatus (final state particle = 1) used for sorting (index 0). Changed  to E later
      'MCParticles.momentum.x',
      'MCParticles.momentum.y',
      'MCParticles.momentum.z',
      'MCParticles.vertex.x',
      'MCParticles.vertex.y',
      'MCParticles.vertex.z',
      'MCParticles.charge',
      'MCParticles.mass',
      'MCParticles.PDG',
      'MCParticles.time',
      # 'MCParticles.simulatorStatus',  <-- Not saved, but used for masking final state particles
  ]
'''
