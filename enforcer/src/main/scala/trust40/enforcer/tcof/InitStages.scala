package trust40.enforcer.tcof

object InitStages extends Enumeration {
  type InitStages = Value

  val ConfigPropagation, VarsCreation, RulesCreation = Value
}
