package ftnn.k4case

import akka.actor.{Actor, Props}
import akka.event.Logging
import ftnn.MarshallersSupport
import ftnn.enforcer.tcof.{AllowAction, NotifyAction}
import spray.json._

import java.io.{FileOutputStream, PrintWriter}
import java.time.LocalDateTime
import java.util.zip.GZIPOutputStream
import scala.collection.mutable

abstract class Permission
case class AllowPermission(subj: String, verb: String, obj: String) extends Permission
case class ComponentNotification(subj: String, action: String, params: List[String])

object Resolver {
  def props(scenarioSpec: ScenarioSpec, traceFileBase: String) = Props(new Resolver(scenarioSpec, traceFileBase))

  case object ResolverReset
  case object ResolverResetDone
  case class ResolverStep(currentTime: LocalDateTime, events: List[ScenarioEvent])
  case class ResolverResult(permissions: List[Permission], notifications: List[ComponentNotification])
}

class Resolver(val scenarioSpec: ScenarioSpec, val traceFileBase: String) extends Actor with MarshallersSupport {
  import Resolver._

  val traceFileWriter = if (traceFileBase != null) new PrintWriter(new GZIPOutputStream(new FileOutputStream(traceFileBase + ".jsonl.gz"))) else null

  private val log = Logging(context.system, this)

  private val solverLimitTime = 60000000000L

  override def postStop(): Unit = {
    if (traceFileWriter != null) {
      traceFileWriter.close()
    }
  }


  private var scenario: Scenario = _

  processReset()

  private def processStep(currentTime: LocalDateTime, events: List[ScenarioEvent]): ResolverResult = {
    log.debug("Resolver started")
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
    val notifs = mutable.ListBuffer.empty[ComponentNotification]
    val factoryTeam = scenario.factoryTeam
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
          case NotifyAction(subj: WithId, notif) =>
            notifs += ComponentNotification(subj.id, notif.getType, notif.getParams)
          case _ =>
        }
      }

    } else {
      log.error("Error. No solution exists.")
    }

    log.info("Resolver finished")


    if (traceFileWriter != null) {
      val dataEntry = JsObject(
        "time" -> currentTime.toJson,
        "workers" -> scenario.workersMap.values.map(_.toJson).toJson,
        "ensembles" -> scenario.factoryTeam.instance.toJson
      )

      traceFileWriter.println(dataEntry.compactPrint)
    }

    ResolverResult(perms.toList, notifs.toList)
  }

  private def processReset(): Unit = {
    scenario = new Scenario(scenarioSpec)
  }

  def receive = {
    case ResolverStep(currentTime, events) =>
      sender() ! processStep(currentTime, events)

    case ResolverReset =>
      processReset()
      sender() ! ResolverResetDone
  }
}

