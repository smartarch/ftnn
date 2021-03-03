package trust40

import akka.actor.ActorSystem
import akka.http.scaladsl.Http
import akka.http.scaladsl.model.StatusCodes._
import akka.http.scaladsl.server.Directives._
import akka.stream.ActorMaterializer
import akka.util.Timeout
import trust40.k4case.{AccessResult, FactoryMap, Simulation, SimulationState, ValidateResult}
import akka.pattern.ask

import scala.concurrent.duration._
import scala.io.StdIn

object Main extends MarshallersSupport {
  def main(args: Array[String]) {
    FactoryMap.init()
    implicit val system = ActorSystem("trust40-enforcer")
    implicit val materializer = ActorMaterializer()
    implicit val executionContext = system.dispatcher

    implicit val timeout = Timeout(1 second)

    val simulation = system.actorOf(Simulation.props())

    // simulation ! Simulation.Play

    val route =
      path("play") {
        post {
          simulation ! Simulation.Play
          complete(OK)
        }
      } ~
      path("pause") {
        post {
          simulation ! Simulation.Pause
          complete(OK)
        }
      } ~
      path("reset") {
        post {
          simulation ! Simulation.Reset
          complete(OK)
        }
      } ~
      path("status") {
        get {
          complete((simulation ? Simulation.Status).mapTo[SimulationState])
        }
      } ~
      path("access" / Segment) { workerId =>
        post {
          complete((simulation ? Simulation.Access(workerId)).mapTo[AccessResult])
        }
      } ~
      path("validate" / Segment / Segment / Segment) { (subjectId, verb, objectId) =>
        post {
          complete((simulation ? Simulation.Validate(subjectId, verb, objectId)).mapTo[ValidateResult])
        }
      }


    val bindingFuture = Http().bindAndHandle(route, "0.0.0.0", 3100)

    println("Listening on 0.0.0.0:3100.")

    /*
    println("Press ENTER to finish.")
    StdIn.readLine()

    bindingFuture
      .flatMap(_.unbind())
      .onComplete(_ => system.terminate())
    */
  }
}
