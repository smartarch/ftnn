package trust40.enforcer.tcof

import scala.collection.mutable

object PrivacyLevel extends Enumeration {
  val PUBLIC = Value("public")
  val INTERNAL_USE = Value("internal_use")
  val HIGHLY_SENSITIVE = Value("highly_sensitive")
  val SENSITIVE = Value("sensitive")
  type PrivacyLevel = Value
}

abstract class PermissionVerb

abstract class Action
case class AllowAction(subj: Component, verb: PermissionVerb, obj: Component) extends Action
case class DenyAction(subj: Component, verb: PermissionVerb, obj: Component, privacyLevel: PrivacyLevel.PrivacyLevel) extends Action
case class NotifyAction(subj: Component, notification: Notification) extends Action

trait WithActionsInEnsemble {
  this: Ensemble =>

  private[tcof] val _actions = mutable.ListBuffer.empty[() => Iterable[Action]]

  private[tcof] def _collectActions(): Iterable[Action] = {
    val groupActions = _ensembleGroups.values.flatMap(group => group.selectedMembers.flatMap(member => member._collectActions()))

    groupActions ++ _actions.flatMap(_())
  }

  def allow(subject: Component, verb: PermissionVerb, objct: Component): Unit = allow(List(subject), verb, List(objct))
  def allow(subjects: => Iterable[Component], verb: PermissionVerb, objct: Component): Unit = allow(subjects, verb, List(objct))
  def allow(subject: Component, verb: PermissionVerb, objects: => Iterable[Component]): Unit = allow(List(subject), verb, objects)

  def allow(subjects: => Iterable[Component], verb: PermissionVerb, objects: => Iterable[Component]): Unit = {
    _actions += (() => {
      for {
        objct <- objects
        subject <- subjects
      } yield AllowAction(subject, verb, objct)
    })
  }

  def deny(subject: Component, verb: PermissionVerb, objct: Component, privacyLevel: PrivacyLevel.PrivacyLevel): Unit = deny(List(subject), verb, List(objct), privacyLevel)
  def deny(subjects: => Iterable[Component], verb: PermissionVerb, objct: Component, privacyLevel: PrivacyLevel.PrivacyLevel): Unit = deny(subjects, verb, List(objct), privacyLevel)
  def deny(subject: Component, verb: PermissionVerb, objects: => Iterable[Component], privacyLevel: PrivacyLevel.PrivacyLevel): Unit = deny(List(subject), verb, objects, privacyLevel)

  def deny(subjects: => Iterable[Component], verb: PermissionVerb, objects: => Iterable[Component], privacyLevel: PrivacyLevel.PrivacyLevel): Unit = {
    _actions += (() => {
      for {
        objct <- objects
        subject <- subjects
      } yield DenyAction(subject, verb, objct, privacyLevel)
    })
  }

  def notify(subject: => Component, notification: Notification): Unit = notifyMany(List(subject), notification)

  def notifyMany(subjects: => Iterable[Component], notification: Notification): Unit = {
    _actions += (() => {
      subjects.foreach(_.notify(notification))
      subjects.map(NotifyAction(_, notification))
    })
  }

  def notifyMany(subjects: => Iterable[Component], notification: Component => Notification): Unit = {
    _actions += (() => {
      subjects.foreach(subj => subj.notify(notification(subj)))
      subjects.map(subj => NotifyAction(subj, notification(subj)))
    })
  }
}
