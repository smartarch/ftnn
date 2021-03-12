package ftnn.enforcer.tcof

trait WithID {
  type IDType

  def id: IDType
}
