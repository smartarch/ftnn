package ftnn

import akka.actor.ActorSystem
import akka.http.scaladsl.Http
import akka.http.scaladsl.model.StatusCodes._
import akka.http.scaladsl.server.Directives._
import akka.pattern.ask
import akka.util.Timeout
import ftnn.k4case.{FactoryMap, Simulation, SimulationState}

import scala.concurrent.duration._
import scala.language.postfixOps

object Main extends MarshallersSupport {
  def main(args: Array[String]) {
    FactoryMap.init()
    implicit val system = ActorSystem("ftnn-simulator")
    implicit val executionContext = system.dispatcher

    implicit val timeout = Timeout(1 second)

    val simulation = system.actorOf(Simulation.props(0, null))

    // simulation ! Simulation.Play

    val route =
      path("play") {
        post {
          simulation ! Simulation.Play(100)
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
      }


    val bindingFuture = Http().newServerAt("0.0.0.0", 3100).bindFlow(route)

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
