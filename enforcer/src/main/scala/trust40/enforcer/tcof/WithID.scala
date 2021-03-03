package trust40.enforcer.tcof

trait WithID {
  type IDType

  def id: IDType
}
