package ftnn.enforcer.tcof

import ftnn.enforcer.tcof.InitStages.InitStages
import ftnn.enforcer.tcof.Utils._
import org.chocosolver.solver.Model

import scala.collection.mutable

trait Component extends WithName with Notifiable {
  override def toString: String =
    s"""Component "$name""""
}
