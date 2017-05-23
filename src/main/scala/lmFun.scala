import breeze.linalg.DenseMatrix
import breeze.plot._
import breeze.linalg.{DenseMatrix, DenseVector}
import breeze.stats.distributions.Gaussian
import breeze.numerics._
import breeze.linalg._
import breeze.stats.regression.leastSquares
import com.github.fommil.netlib.BLAS._
import org.jfree.chart.annotations.XYTextAnnotation
import com.github.fommil.netlib.BLAS.{ getInstance => blas }
import com.github.fommil.netlib.LAPACK.{ getInstance => lapack }

/**
  * Created by antonia on 16/03/17.
  * This program is for simple linear regression. There are 2 different implementations. The simple OLS by using the leastSquares function and the one with QR factorisation that uses backwards substitution via the backsolved function as defined below.
  * The code is based on: https://github.com/darrenjw/statslang-scala/tree/master/regression/src/main/scala/regression
  * The data generation is based on: https://darrenjw.wordpress.com/2017/02/08/a-quick-introduction-to-apache-spark-for-statisticians/
  */
object lmFun {
  //Simulate the x1 and x2 covariates and the residuals and save them in a denseMatrix.
  //Returns a tuple with the denseVector y (the simulated observed values and the denseMatrix xx with the intercept and the values of x1 and x2
  def dataSimulation(sampleSize:Int)={
    val x1=DenseVector(breeze.stats.distributions.Gaussian(1.0,2.0).sample(sampleSize).toArray) //first covariate    /OR in 2 steps val g = breeze.stats.distributions.Gaussian(1.0,2.0)
    val x2=DenseVector(breeze.stats.distributions.Gaussian(0.0,1.0).sample(sampleSize).toArray) //second covariate
    val eps = DenseVector(breeze.stats.distributions.Gaussian(0.0,1.0).sample(sampleSize).toArray)//residual noise

    //Create the matrix xx which will be (1,x1,x2) in order to be multiplied with (1.5,2,1) and then added to eps to get the observations.
    val xx= DenseMatrix.zeros[Double](sampleSize,3)
    xx(::,0):= DenseVector.ones[Double](sampleSize)
    xx(::,1):= x1
    xx(::,2):= x2
    val beta=DenseVector(1.5,2.0,1.0)
    val y= xx*beta + eps
    (y,xx)
  }

  // OLS

  //val fit = leastSquares(xx,y)
  // Returns a tuple3 with the estimated coefficients β0,β1,β2 , a denseVector with the fittedY and a denseVector with the residuals.
  def ols(ab: (DenseVector[Double],DenseMatrix[Double]))={
    val (y,x)=ab
    val fit = leastSquares(x,y)
    val fittedY= fit.coefficients(0):+ ((fit.coefficients(1):*x(::,1)) + (fit.coefficients(2):*x(::,2))) //Extra brackets for the :* to make it work!!
    val residuals = y-fittedY
    val (coeffs, fittedValY,resid)= (fit.coefficients, fittedY, residuals)
    (coeffs, fittedValY,resid)
  }

  //Linear Regression QR factorisation
  def lmQR(ab: (DenseVector[Double],DenseMatrix[Double]))={
    val (y,x)=ab
    // 1st way. Return the betaCoeffs
    //val betaCoeffs= x\y

    val QR = qr.reduced(x) //we use the reduced to get the correct results. It is nessecary for thin matrices. Just qr gives wrong results. This tinnedQR or reducedQR decomposes A to a thin Q and a R. Just qr decomposes A to a big Q and a thin R
    val Q = QR.q
    val R = QR.r
    val Qty = Q.t * y //Rβ=Qty and then backsolve
    val coeffs = backSolve(R, Qty)
    val fitted = Q * Qty
    val residuals = y - fitted
    (coeffs, fitted, residuals)
  }


  //For univariate
  // A: DenseMatrix[Double] is the Upper triangular R
  def backSolve(A: DenseMatrix[Double], y: DenseVector[Double]): DenseVector[Double] = {
    val yc = y.copy
    // ARGUMENTS OF blas.dtrsv
    // 'U' or 'u'   A is an upper triangular matrix
    // TRANS = 'N' or 'n'   A*x = b
    // DIAG = 'N' or 'n'   A is not assumed to be unit triangular
    // N specifies the order of the matrix A
    // A is DOUBLE PRECISION array of DIMENSION
    // On entry, LDA specifies the first dimension of A as declared in the calling (sub) program
    // X is DOUBLE PRECISION array of dimension
    // specifies the increment for the elements of X
    blas.dtrsv("U", "N", "N", A.cols, A.toArray, A.rows, yc.data, 1)
    yc
  }


  // Function to plot the residuals vs the fitted values. Arguments: tuple2 with the fitted values and the residuals.
  def plotResiduals(ab: (DenseVector[Double],DenseVector[Double])): Figure = {
    val f = Figure()
    val p = f.subplot(0)
    val (fittedY,residuals)=ab
    p += plot(fittedY, residuals, '.')
    p.xlabel = "Fitted Values"
    p.ylabel = "Residuals"
    p.title = "Residuals against fitted values"
    val p2 = f.subplot(1, 2, 1)
    p2 += hist(residuals)
    p2.title = "Residual Histogram"
    f
  }

  def main(args: Array[String]) {
    val n=1000
    val (simData,x)=dataSimulation(n)
    //Simple OLS
    //val (coeffs, fittedY,residuals)= ols(simData,x)
    //println(coeffs)
    // Linear Regression with QR
    val (coeffsQR, fittedYQR,residualsQR)= lmQR(simData,x)
    println(coeffsQR)
    plotResiduals(fittedYQR,residualsQR)
  }
}
