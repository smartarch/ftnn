package trust40.enforcer.tcof

import trust40.enforcer.tcof.InitStages.InitStages

trait Initializable {
  private[tcof] def _init(stage: InitStages, config: Config): Unit = {
  }
}



