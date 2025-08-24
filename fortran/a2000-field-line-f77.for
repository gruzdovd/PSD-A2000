*  A2012 - ver.3.2
* needed in a2012-v3_3.for

      SUBROUTINE a2000_main(X1, X2, X3, X4, X5, X6, X7,
     *                      ut, iy, mo, id, ro, v, bimf, dst, al,
     *                      x, sl, h)
      DIMENSION bimf(3), x(3), output_array(9,100000)
c	  INTEGER, INTENT(IN) :: X1, X2, X3, X4, X5, X6, X7
c      REAL, INTENT(IN) :: ut, ro, v, dst, al, sl, h
c      REAL, INTENT(IN) :: bimf(3), x(3)      
c	  INTEGER num_points
      COMMON /OUTPUT_DATA/ output_array, num_points
	  num_points = 0
      call pstatus(X1, X2, X3, X4, X5, X6, X7)
      call a2000f_line(ut, iy, mo, id, ro, v, bimf, dst, al,
     *                x, sl, h, 1)
      
      RETURN
      END
	  
      subroutine a2000f_line(ut,iy,mo,id,ro,v,bimf,dst,al,x0,sl,h,kpr)
*-----------------------------------------------------------------
*  Calculation of magnetic field line in GSM coordinates
*  from point x(3) at time moment UT (Universal Time) on
*  year=iy;
*  day=id in month=mo.
*  ro, V, bimf - are solar wind density and velocity, and IMF.
*  dst - value of Dst index;
*  AL  - value of al index. 
*  bm=0 when point x(3) is outside the magnetosphere
*                    sl, h  - the maximum length of the magnetic field line     
*                             and step along the field line, Re;                
*                    kpr -    provides the control of the field line printing   
*                             (1 - enabled, 0 - disabled).                      
* NOTE: The resulting field line is writing in file output.dat                  
*       in the working directory.                                               
* WARNING: Because of the paraboloid coordinates singularity, avoid             
*          the magnetic field calculations at the Ox axis and magnetic          
*          field line calculations on the y=0 plane.                            
*                                                                               
* Written by V.Kalegaev                                                         
*-------------------------------------------------------------
      COMMON /OUTPUT_DATA/ output_array, num_points
	  DIMENSION output_array(9, 100000)
      INTEGER num_points
      DIMENSION x0(3), bimf(3), bm(3), par(10), bdd(7,3)

      call submod(ut,iy,mo,id,ro,v,bimf,dst,al,par)

      r1=par(6)
       IF(x0(1)+0.5/R1*(x0(2)**2+x0(3)**2).GT.R1) then
       do i=1,3
       bm(i)=0.
       do j=1,7
       bdd(j,i)=0.
       end do
       end do
       return
       end if
        call P_line(x0,par,sl,h,kpr)
c        call Pi_line(mlt,lat,par,sl,h,kpr)
125   format(5x,'x',5x,'y',5x,'z',10x,'AB',8x,'Bx',8x,'By',8x,'Bz',/)
126   format(2x,3f6.2,2x,4f10.1)
      return
      END
      subroutine a2000f_line_i(ut,iy,mo,id,ro,v,bimf,dst,al,amlt,
     *alat,sl,h,kpr)
*-----------------------------------------------------------------
*  Calculation of magnetic field line in GSM coordinates
*  from point x(3) at time moment UT (Universal Time) on
*  year=iy;
*  day=id in month=mo.
*  ro, V, bimf - are solar wind density and velocity, and IMF.
*  dst - value of Dst index;
*  AL  - value of al index. 
*  bm=0 when point x(3) is outside the magnetosphere
*                    sl, h  - the maximum length of the magnetic field line     
*                             and step along the field line, Re;                
*                    kpr -    provides the control of the field line printing   
*                             (1 - enabled, 0 - disabled).                      
* NOTE: The resulting field line is writing in file output.dat                  
*       in the working directory.                                               
* WARNING: Because of the paraboloid coordinates singularity, avoid             
*          the magnetic field calculations at the Ox axis and magnetic          
*          field line calculations on the y=0 plane.                            
*                                                                               
* Written by V.Kalegaev                                                         
*-------------------------------------------------------------
      COMMON /OUTPUT_DATA/ output_array, num_points
	  DIMENSION output_array(9, 100000)
      INTEGER num_points 
      DIMENSION x0(3), bimf(3), bm(3), par(10), bdd(7,3)

c      print *, ut,iy,mo,id,ro,v,bimf,dst,al

      call submod(ut,iy,mo,id,ro,v,bimf,dst,al,par)
c      print *, ut,iy,mo,id,ro,v,bimf,dst,al
c      print *, par
      r1=par(6)
       IF(x0(1)+0.5/R1*(x0(2)**2+x0(3)**2).GT.R1) then
       do i=1,3
       bm(i)=0.
       do j=1,7
       bdd(j,i)=0.
       end do
       end do
       return
       end if
c        call P_line(x0,par,sl,h,kpr)
        call Pi_line(amlt,alat,par,sl,h,kpr)
125   format(5x,'x',5x,'y',5x,'z',10x,'AB',8x,'Bx',8x,'By',8x,'Bz',/)
126   format(2x,3f6.2,2x,4f10.1)
      return
      END
