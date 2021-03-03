package trust40.enforcer.tcof

import trust40.enforcer.tcof.InitStages.InitStages
import trust40.enforcer.tcof.Utils._
import org.chocosolver.solver.Model

import scala.collection.mutable

trait Component extends WithName with Notifiable {
  override def toString: String =
    s"""Component "$name""""
}
