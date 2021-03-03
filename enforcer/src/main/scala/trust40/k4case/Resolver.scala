package trust40.k4case

import java.time.LocalDateTime

import akka.actor.{Actor, Props}
import akka.event.Logging
import trust40.enforcer.sdq.DesignTimeDecisionMakerImpl
import scala.collection.JavaConverters._
import trust40.enforcer.tcof.PrivacyLevel.PrivacyLevel
import trust40.enforcer.tcof.{AllowAction, DenyAction, NotifyAction}
import trust40.k4case.Simulation.Reset


import scala.collection.mutable

abstract class Permission
case class AllowPermission(subj: String, verb: String, obj: String) extends Permission
case class DenyPermission(subj:String, ver:String,obj:String, lvl:PrivacyLevel) extends Permission
case class ComponentNotification(subj: String, action: String, params: List[String])

object Resolver {
  def props(scenarioSpec: TestScenarioSpec) = Props(new Resolver(scenarioSpec))

  final case class Resolve(currentTime: LocalDateTime, events: List[ScenarioEvent])
  final case class ResolverResult(epoch: Int, permissions: List[Permission], notifications: List[ComponentNotification], rejectedPermissions: Map[AllowPermission, List[DenyPermission]])
}

class Resolver(val scenarioSpec: TestScenarioSpec) extends Actor {
  import Resolver._

  private val log = Logging(context.system, this)

  private val solverLimitTime = 60000000000L

  private var scenario: TestScenario = _

  private var currentEpoch: Int = _


  private def processResolve(currentTime: LocalDateTime, events: List[ScenarioEvent]): Unit = {
    // log.info("Resolver started")
    // log.info("Time: " + currentTime)
    // log.info("Events: " + events)


    scenario.now = currentTime

    for (event <- events) {
      val worker = scenario.workersMap(event.person)
      worker.position = event.position

      if (event.eventType == "retrieve-head-gear") {
        worker.hasHeadGear = true
      }

      if (event.eventType == "return-head-gear") {
        worker.hasHeadGear = false
      }
    }

    val perms = mutable.ListBuffer.empty[AllowPermission]
    val deny = mutable.ListBuffer.empty[DenyPermission]
    val notifs = mutable.ListBuffer.empty[ComponentNotification]
    var networkPermission: List[AllowPermission] = null;
    val factoryTeam = scenario.factoryTeam
    val rejectedRules = new java.util.HashMap[AllowPermission, java.util.List[DenyPermission]];
    factoryTeam.init()
    factoryTeam.solverLimitTime(solverLimitTime)
    factoryTeam.solve()
        if (factoryTeam.exists) {
      // log.info("Utility: " + shiftTeams.instance.solutionUtility)
      // log.info(shiftTeams.instance.toString)

      factoryTeam.commit()

      for (action <- factoryTeam.actions) {
        // println(action)
        action match {
          case AllowAction(subj: WithId, verb, obj: WithId) =>
            perms += AllowPermission(subj.id, verb.toString, obj.id)
          case DenyAction(subj: WithId, verb, obj: WithId, privacyLevel) =>
            deny += DenyPermission(subj.id, verb.toString, obj.id, privacyLevel)
          case NotifyAction(subj: WithId, notif) =>
            notifs += ComponentNotification(subj.id, notif.getType, notif.getParams)
          case _ =>
        }
      }
      val rules = new DesignTimeDecisionMakerImpl
      networkPermission = rules.validatePolicies(perms.toList.asJava, deny.toList.asJava, rejectedRules).asScala.toList
      // call to KIT <- factoryTeam.actions ... => consistent set of action
    } else {
      log.error("Error. No solution exists.")
    }

    // log.info("Resolver finished")

    val rejectedRulesScala = (for ((allowPermission, denyRules) <- rejectedRules.asScala) yield allowPermission -> denyRules.asScala.toList).toMap

    sender() ! ResolverResult(currentEpoch, networkPermission, notifs.toList, rejectedRulesScala)
  }

  private def processReset(epoch: Int): Unit = {
    currentEpoch = epoch
    scenario = new TestScenario(scenarioSpec)
  }

  def receive = {
    case Resolve(currentTime, events) =>
      processResolve(currentTime, events)

    case Reset(epoch: Int) =>
      processReset(epoch)
  }

}

