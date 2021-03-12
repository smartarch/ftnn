package ftnn

import akka.actor.{ActorRef, ActorSystem}
import akka.event.{LogSource, Logging}
import akka.pattern.ask
import akka.util.Timeout
import ftnn.k4case.{FactoryMap, Simulation, SimulationState}

import java.io.File
import scala.concurrent.Await
import scala.concurrent.duration._
import scala.language.postfixOps

object TraceRecord {
  def main(args: Array[String]) {
    FactoryMap.init()
    implicit val system = ActorSystem("ftnn-simulator")
    implicit val executionContext = system.dispatcher

    implicit val timeout = Timeout(60 second)

    implicit val logSource = new LogSource[TraceRecord.type] {
      def genString(x: TraceRecord.type) = "Main"
    }

    val log = Logging(system, this)

    val concurrencyLevel = 32
    val iterationStart = args(0).toInt
    val iterationUntil = args(1).toInt

    var iterationsToGo = (iterationStart until iterationUntil).toList
    var iterationsInProgress = List.empty[(Int, ActorRef)]

    while (!iterationsToGo.isEmpty || !iterationsInProgress.isEmpty) {
      var newIterationsInProgress = List.empty[(Int, ActorRef)]
      for ((iterIdx, sim) <- iterationsInProgress.reverse) {
        val state = Await.result((sim ? Simulation.Status).mapTo[SimulationState], timeout.duration)
        log.info(s"Iteration ${iterIdx}: " + state.time.toString)

        if (state.playState != Simulation.State.END) {
          newIterationsInProgress = ((iterIdx, sim)) :: newIterationsInProgress
        } else {
          system.stop(sim)
        }
      }

      iterationsInProgress = newIterationsInProgress

      while (iterationsInProgress.size < concurrencyLevel && !iterationsToGo.isEmpty) {
        val iterIdx = iterationsToGo.head
        iterationsToGo = iterationsToGo.tail

        val dirPath = f"traces/v1/${iterIdx / 1000}%04d"

        (new File(dirPath)).mkdirs()

        val sim = system.actorOf(Simulation.props(iterIdx, f"${dirPath}/${iterIdx % 1000}%03d"))
        sim ! Simulation.Play(0)

        iterationsInProgress = ((iterIdx, sim)) :: iterationsInProgress
      }

      Thread.sleep(1000)
    }

    system.terminate()
  }
}
