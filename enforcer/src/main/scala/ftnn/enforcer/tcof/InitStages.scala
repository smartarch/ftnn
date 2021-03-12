package ftnn.enforcer.tcof

object InitStages extends Enumeration {
  type InitStages = Value

  val ConfigPropagation, VarsCreation, RulesCreation = Value
}
