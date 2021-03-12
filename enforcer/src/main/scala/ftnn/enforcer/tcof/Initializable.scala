package ftnn.enforcer.tcof

import ftnn.enforcer.tcof.InitStages.InitStages

trait Initializable {
  private[tcof] def _init(stage: InitStages, config: Config): Unit = {
  }
}



